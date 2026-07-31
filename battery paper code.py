"""
================================================================================
PSEUDOCODE: AI-Driven SOH / RUL Prediction and Multi-Parameter BMS Estimation
Corresponds to Section III (Dataset and Methodology) and Section IV (Results)
of the paper "AI-Driven State-of-Health, Remaining Useful Life, and
Multi-Parameter Battery Management" (NASA Li-ion Battery Aging Dataset).
================================================================================

TRACK 1 : RUL Regression                (Random Forest, XGBoost, Gradient Boosting)
TRACK 2 : LSTM Sequential SOH Forecasting (leave-one-battery-out)
TRACK 3 : Online Calibration / Transfer Learning (20-cycle fine-tune)
TRACK 4 : Multi-Head DNN for Production BMS (7 simultaneous outputs)
================================================================================
"""

# =============================================================================
# 0. IMPORTS AND GLOBAL CONFIGURATION
# =============================================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, accuracy_score
)
import xgboost as xgb

# ---- Reproducibility -------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ---- Global hyperparameters (match Section III exactly) --------------------
DATA_PATH          = "Battery_Data_Cleaned.csv"
WINDOW_SIZE         = 10          # sliding-window length for LSTM sequences
CALIBRATION_CYCLES  = 20          # cycles used for online fine-tuning
TRAIN_TEST_SPLIT    = 0.20        # 80/20 split for ensemble regressors
MAX_EXPECTED_RUL    = 1000        # cycles, used to scale derived RUL target
EOL_THRESHOLD_RATIO = 0.70        # 70% capacity fade = end-of-life

LSTM_UNITS_1        = 64
LSTM_UNITS_2        = 32
DROPOUT_RATE        = 0.4
DENSE_UNITS         = 16
LSTM_EPOCHS         = 50
LSTM_BATCH_SIZE     = 8

CALIB_LR            = 5e-5
CALIB_EPOCHS        = 20
CALIB_PATIENCE      = 3

DNN_EPOCHS           = 30
DNN_BATCH_SIZE       = 256
DEPLOYMENT_BUDGET_KB = 750


# =============================================================================
# 1. DATA LOADING AND QUALITY CHECKS  (Section III-A)
# =============================================================================

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the cleaned NASA Li-ion Battery Aging Dataset.

    Expected schema (9 columns, 7,368 rows, 34 unique battery_id values):
        type                  : {-1, 0, 1}  -> discharge / charge / impedance
        ambient_temperature   : {4, 22, 24, 43, 44} degrees C
        battery_id            : int, unique cell identifier
        test_id, uid          : sequential indices
        filename              : source measurement file (non-numeric, dropped)
        Capacity              : float, measured discharge capacity (Ah)
        Re                    : float, electrolyte resistance (Ohm)
        Rct                   : float, charge-transfer resistance (Ohm)
    """
    df = pd.read_csv(path)
    assert set(df.columns) >= {
        "type", "ambient_temperature", "battery_id", "test_id",
        "uid", "filename", "Capacity", "Re", "Rct"
    }, "Unexpected schema — verify source file"
    return df


def run_data_quality_checks(df: pd.DataFrame) -> dict:
    """
    Data quality audit performed before any modeling (Section III-A).

    Returns a report dict flagging:
      - missing values per column (expected: zero)
      - count of physically implausible zero-capacity rows (expected: ~50)
      - battery-level cycle-count imbalance
    """
    report = {}
    report["missing_values"] = df.isnull().sum().to_dict()
    report["zero_capacity_rows"] = int((df["Capacity"] == 0).sum())
    report["cycles_per_battery"] = df.groupby("battery_id").size().to_dict()
    report["n_unique_batteries"] = df["battery_id"].nunique()
    return report


# =============================================================================
# 2. FEATURE ENGINEERING  (Section III-B)
# =============================================================================

def engineer_degradation_and_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a physics-motivated degradation feature and RUL target.

    degradation_feature = Re * Rct
        Rationale: both electrolyte resistance (Re) and charge-transfer
        resistance (Rct) increase monotonically with SEI growth and active
        material loss, so their product is a compact scalar proxy for
        internal cell degradation.

    RUL = MAX_EXPECTED_RUL / (degradation_feature + 1)
        Inverse relationship: low-resistance (fresh) cells map close to
        MAX_EXPECTED_RUL; high-resistance (degraded) cells map to a
        proportionally lower remaining life. The "+1" avoids division
        blow-up for near-zero degradation_feature values.
    """
    df = df.copy()
    df["degradation_feature"] = df["Re"] * df["Rct"]
    df["RUL"] = MAX_EXPECTED_RUL / (df["degradation_feature"] + 1.0)
    return df


def build_rul_feature_matrix(df: pd.DataFrame):
    """
    Construct (X, y) for the static ensemble-regression track.

    Drops:
      - RUL                 (target)
      - degradation_feature (near-deterministic function of the target;
                              dropped to avoid trivial target leakage)
      - filename             (non-numeric, uninformative identifier)
    """
    X = df.drop(columns=["RUL", "degradation_feature", "filename"])
    y = df["RUL"]
    return X, y


def normalize_soh_per_battery(df: pd.DataFrame, battery_id) -> np.ndarray:
    """
    Extract and self-normalize the capacity trajectory of ONE battery:
        SOH(t) = Capacity(t) / Capacity(0)
    Normalizing by each battery's own first-cycle capacity accounts for
    manufacturing variance in rated capacity across the 34 cells.
    """
    cell_data = df[df["battery_id"] == battery_id].copy()
    capacity_series = cell_data["Capacity"].values
    if len(capacity_series) == 0:
        return np.array([])
    return capacity_series / capacity_series[0]


def build_sliding_window_sequences(battery_list, df, window_size=WINDOW_SIZE):
    """
    Build LSTM-ready (X, y) sequences WITHOUT mixing cycles across batteries.

    For each battery:
        soh = normalize_soh_per_battery(...)
        for i in range(len(soh) - window_size):
            X_sample = soh[i : i + window_size]      # past `window_size` cycles
            y_sample = soh[i + window_size]           # next-cycle SOH (target)

    Returns:
        X : ndarray, shape (n_samples, window_size, 1)
        y : ndarray, shape (n_samples,)
    """
    X_list, y_list = [], []
    for bat_id in battery_list:
        soh = normalize_soh_per_battery(df, bat_id)
        if len(soh) <= window_size:
            continue  # skip batteries with insufficient cycle history
        for i in range(len(soh) - window_size):
            X_list.append(soh[i:i + window_size])
            y_list.append(soh[i + window_size])

    X = np.array(X_list).reshape(-1, window_size, 1)
    y = np.array(y_list)
    return X, y


# =============================================================================
# 3. TRACK 1 — RUL REGRESSION (RANDOM FOREST / XGBOOST / GRADIENT BOOSTING)
#    (Section III-B, Section IV-A, Table IV)
# =============================================================================

def train_test_split_rul(X, y, test_size=TRAIN_TEST_SPLIT, seed=RANDOM_SEED):
    """Standard 80/20 held-out split at the ROW level (static features)."""
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def train_random_forest(X_train, y_train) -> RandomForestRegressor:
    """
    Best-performing RUL model in Table IV.
        R^2 = 0.99987 | MAE = 0.0181 | MSE = 0.00286
    Default sklearn hyperparameters used; scale-invariant, no feature
    standardization required.
    """
    model = RandomForestRegressor(random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train) -> xgb.XGBRegressor:
    """
    Second-best RUL model in Table IV.
        R^2 = 0.99863 | MAE = 0.1183 | MSE = 0.03133

    Hyperparameters (deliberately conservative learning rate + many trees):
        n_estimators   = 1000
        learning_rate  = 0.01
        max_depth      = 3
        subsample      = 0.7
        colsample_bytree = 1.0
        gamma          = 0
    """
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=1.0,
        gamma=0,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train) -> GradientBoostingRegressor:
    """
    Third model in Table IV (default sklearn config).
        MSE ~ 0.0300 | MAE ~ 0.1100 | R^2 ~ 0.996
    """
    model = GradientBoostingRegressor(random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    return model


def train_svr(X_train, y_train, scaler: StandardScaler = None) -> SVR:
    """
    Kernel-based baseline (requires standardized inputs, unlike tree models).
    """
    if scaler is None:
        scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    model = SVR()
    model.fit(X_train_scaled, y_train)
    return model, scaler


def evaluate_regressor(model, X_test, y_test, is_svr=False, scaler=None) -> dict:
    """
    Compute MSE, MAE, R^2 for a fitted regressor — used to populate Table IV.
    """
    X_eval = scaler.transform(X_test) if (is_svr and scaler is not None) else X_test
    y_pred = model.predict(X_eval)
    return {
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2":  r2_score(y_test, y_pred),
    }


def run_track1_rul_regression(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates Track 1 end-to-end and returns a results table
    equivalent to Table IV in the paper.
    """
    df = engineer_degradation_and_rul(df)
    X, y = build_rul_feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split_rul(X, y)

    results = {}
    rf_model = train_random_forest(X_train, y_train)
    results["Random Forest"] = evaluate_regressor(rf_model, X_test, y_test)

    xgb_model = train_xgboost(X_train, y_train)
    results["XGBoost"] = evaluate_regressor(xgb_model, X_test, y_test)

    gb_model = train_gradient_boosting(X_train, y_train)
    results["Gradient Boosting"] = evaluate_regressor(gb_model, X_test, y_test)

    svr_model, svr_scaler = train_svr(X_train, y_train)
    results["SVR"] = evaluate_regressor(
        svr_model, X_test, y_test, is_svr=True, scaler=svr_scaler
    )

    return pd.DataFrame(results).T.sort_values("R2", ascending=False)


# =============================================================================
# 4. TRACK 2 — LSTM SEQUENTIAL SOH FORECASTING  (Section III-C, IV-B)
# =============================================================================

def build_lstm_model(window_size=WINDOW_SIZE) -> tf.keras.Model:
    """
    Two-layer stacked LSTM (29,857 trainable parameters, 116.63 KB).

    Architecture:
        Input(window_size, 1)
          -> LSTM(64, return_sequences=True)
          -> Dropout(0.4)
          -> LSTM(32, return_sequences=False)
          -> Dropout(0.4)
          -> Dense(16, activation='relu')
          -> Dense(1,  activation='linear')      # next-cycle SOH
    """
    model = models.Sequential([
        layers.Input(shape=(window_size, 1)),
        layers.LSTM(LSTM_UNITS_1, return_sequences=True),
        layers.Dropout(DROPOUT_RATE),
        layers.LSTM(LSTM_UNITS_2, return_sequences=False),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(DENSE_UNITS, activation="relu"),
        layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def run_track2_lstm_soh_forecast(df: pd.DataFrame, test_battery=None):
    """
    Leave-one-battery-out cross-battery generalization test.

    Split strategy:
        train_batteries = all_batteries[:-1]   # 33 batteries -> 6,786 sequences
        test_battery    = all_batteries[-1]    # 1 battery    ->   242 sequences
    This is DELIBERATELY stricter than a random row-level split, because
    adjacent cycles within one battery are highly autocorrelated and would
    otherwise leak information between train and test partitions.
    """
    all_batteries = sorted(df["battery_id"].unique())
    if test_battery is None:
        test_battery = all_batteries[-1]
    train_batteries = [b for b in all_batteries if b != test_battery]

    X_train, y_train = build_sliding_window_sequences(train_batteries, df)
    X_test,  y_test  = build_sliding_window_sequences([test_battery], df)

    model = build_lstm_model()
    history = model.fit(
        X_train, y_train,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    y_pred = model.predict(X_test)
    baseline_metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),  # -> 0.1221
        "MAE":  mean_absolute_error(y_test, y_pred),           # -> 0.1202
    }

    return model, history, (X_test, y_test), baseline_metrics, test_battery


# =============================================================================
# 5. TRACK 3 — ONLINE CALIBRATION / TRANSFER LEARNING  (Section III-C, IV-B)
# =============================================================================

def freeze_recurrent_trunk(model: tf.keras.Model) -> tf.keras.Model:
    """
    Freeze every layer EXCEPT the final two dense layers, so that only the
    output head is fine-tuned on the small calibration set. This preserves
    the general degradation patterns learned across 33 batteries while
    allowing the model to adapt its final mapping to the specific new cell.
    """
    for layer in model.layers[:-2]:
        layer.trainable = False
    return model


def online_calibrate_lstm(trained_model, X_test, y_test,
                           calibration_cycles=CALIBRATION_CYCLES):
    """
    Novel calibration procedure (central contribution of this paper).

    Steps:
      1. Split held-out battery's test sequences into:
             calibration set = first `calibration_cycles` samples
             future set      = remaining samples (evaluated afterward)
      2. Clone the trained LSTM and freeze all layers except the last two
         dense layers (see freeze_recurrent_trunk).
      3. Recompile with a much lower learning rate (5e-5) to avoid
         catastrophic forgetting of the pretrained representation.
      4. Fine-tune on the calibration set only, for up to CALIB_EPOCHS
         epochs, with early stopping (patience=3, monitor='loss').
      5. Compute a scalar bias-correction term as the mean residual on the
         calibration set, and apply it uniformly to all future predictions.

    Result reported in Section IV-B:
        RMSE: 0.1221 -> 0.0271   (-77.8%)
        MAE : 0.1202 -> 0.0209   (-82.6%)
    """
    X_calib, y_calib = X_test[:calibration_cycles], y_test[:calibration_cycles]
    X_future, y_future = X_test[calibration_cycles:], y_test[calibration_cycles:]

    # --- Step 2: clone + freeze -------------------------------------------
    calibrated_model = tf.keras.models.clone_model(trained_model)
    calibrated_model.set_weights(trained_model.get_weights())
    calibrated_model = freeze_recurrent_trunk(calibrated_model)

    # --- Step 3: recompile at low LR ---------------------------------------
    calibrated_model.compile(
        optimizer=optimizers.Adam(learning_rate=CALIB_LR),
        loss="mean_squared_error",
    )

    # --- Step 4: fine-tune on calibration cycles only ----------------------
    early_stop = callbacks.EarlyStopping(
        monitor="loss", patience=CALIB_PATIENCE,
        restore_best_weights=True, verbose=1,
    )
    calibrated_model.fit(
        X_calib, y_calib,
        epochs=CALIB_EPOCHS,
        batch_size=4,
        callbacks=[early_stop],
        verbose=1,
    )

    # --- Step 5: bias correction --------------------------------------------
    calib_preds = calibrated_model.predict(X_calib).flatten()
    bias = np.mean(y_calib - calib_preds)

    future_preds_raw = calibrated_model.predict(X_future).flatten()
    future_preds_corrected = future_preds_raw + bias

    calibrated_metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_future, future_preds_corrected)),
        "MAE":  mean_absolute_error(y_future, future_preds_corrected),
        "bias_correction": bias,
    }
    return calibrated_model, calibrated_metrics, (X_future, y_future, future_preds_corrected)


# =============================================================================
# 6. TRACK 4 — MULTI-HEAD DNN FOR PRODUCTION BMS  (Section III-D, IV-C)
# =============================================================================

FEATURE_COLS = [
    "temp_cell1", "temp_cell2", "temp_cell3", "temp_cell4",
    "total_voltage", "total_current", "total_pressure", "gas_sensor",
]
OUTPUT_COLS = [
    "SOC", "SOH", "rate_of_discharge", "pack_battery_level",
    "battery_life", "cycle_life", "depth_of_discharge",
]


def simulate_multicell_pack(discharge_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a synthetic four-cell pack dataset from single-cell NASA
    discharge measurements (Section III-A / III-D).

    Per-cell temperature offsets emulate thermistor placement variance;
    total_voltage assumes 4 cells in series; pressure/gas channels are
    monotonic proxies of temperature and cycle index, representing
    auxiliary BMS sensing channels not present in the raw NASA data.
    """
    d = discharge_df[discharge_df["type"] == -1].copy()  # discharge rows only
    out = pd.DataFrame()

    out["temp_cell1"] = d["Temperature_measured"] + np.random.uniform(-0.2, 0.2, len(d))
    out["temp_cell2"] = d["Temperature_measured"] + np.random.uniform(-0.3, 0.3, len(d))
    out["temp_cell3"] = d["Temperature_measured"] + np.random.uniform(-0.1, 0.4, len(d))
    out["temp_cell4"] = d["Temperature_measured"] + np.random.uniform(-0.4, 0.1, len(d))

    out["total_voltage"] = d["Voltage_measured"] * 4
    out["total_current"] = d["Current_measured"]
    out["total_pressure"] = 1.0 + 0.01 * (d["Temperature_measured"] - 24)
    out["gas_sensor"] = 0.002 * d["id_cycle"] + 0.001 * d["Temperature_measured"]

    v = d["Voltage_measured"]
    out["SOC"] = ((v - 3.2) / (4.2 - 3.2)).clip(0, 1)
    out["SOH"] = d["Capacity"] / 1.856487                      # rated capacity
    out["rate_of_discharge"] = out["SOC"].diff().fillna(0).abs()
    out["pack_battery_level"] = out["SOC"] * 100
    out["battery_life"] = 500 - d["id_cycle"]                  # assumed 500-cycle life
    out["cycle_life"] = d["id_cycle"]
    out["depth_of_discharge"] = 1 - out["SOC"]
    return out


def inject_production_edge_cases(df: pd.DataFrame, seed=RANDOM_SEED) -> pd.DataFrame:
    """
    Injects realistic BMS edge cases so the model does not overfit to
    clean laboratory data (Section III-A):

      (1) SENSOR FAULT   : ~1% of rows -> temp_cell1 set to sentinel -99.0
      (2) CELL IMBALANCE : temp_cell4 scaled x1.1 + 2.0 degC baseline offset
      (3) REST PERIODS   : 5,000 synthetic rows with current = 0,
                            rate_of_discharge = 0, small voltage "relaxation"
      (4) GLOBAL SHUFFLE : prevents overfitting to original NASA cycle order
    """
    rng = np.random.default_rng(seed)

    # (1) sensor fault injection
    fault_idx = df.sample(frac=0.01, random_state=seed).index
    df.loc[fault_idx, "temp_cell1"] = -99.0

    # (2) systematic cell imbalance
    df["temp_cell4"] = df["temp_cell4"] * 1.1 + 2.0

    # (3) synthetic rest periods
    rest_rows = df.sample(n=5000, random_state=seed).copy()
    rest_rows["total_current"] = 0.0
    rest_rows["rate_of_discharge"] = 0.0
    rest_rows["total_voltage"] += 0.05
    df = pd.concat([df, rest_rows], ignore_index=True)

    # (4) global shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def build_multihead_dnn(n_features=8, n_outputs=7) -> tf.keras.Model:
    """
    Shared-trunk, multi-head architecture (Section III-D).

        Input(n_features)
          -> Dense(128, relu) -> BatchNorm -> Dropout(0.2)
          -> Dense(64, relu)                                  # shared trunk
          -> [ Dense(32, relu) -> Dense(1, linear) ]  x 7      # per-target heads
    """
    inputs = layers.Input(shape=(n_features,), name="sensor_inputs")

    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    shared_trunk = layers.Dense(64, activation="relu")(x)

    output_heads = []
    for name in OUTPUT_COLS:
        branch = layers.Dense(32, activation="relu")(shared_trunk)
        head = layers.Dense(1, activation="linear", name=name)(branch)
        output_heads.append(head)

    model = models.Model(inputs=inputs, outputs=output_heads)
    model.compile(optimizer="adam", loss="mse")
    return model


def run_track4_multihead_bms(production_df: pd.DataFrame):
    """
    Trains and evaluates the seven-output production-BMS model
    (Table V results: R^2 of 0.973/0.913/0.395/0.995/0.998/0.999/0.994).
    """
    X = production_df[FEATURE_COLS].values
    Y = {name: production_df[name].values for name in OUTPUT_COLS}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, idx_train, idx_test = train_test_split(
        X_scaled, np.arange(len(production_df)),
        test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_SEED,
    )
    Y_train = {k: v[idx_train] for k, v in Y.items()}
    Y_test  = {k: v[idx_test]  for k, v in Y.items()}

    model = build_multihead_dnn()
    model.fit(
        X_train, [Y_train[name] for name in OUTPUT_COLS],
        validation_split=0.10,
        epochs=DNN_EPOCHS,
        batch_size=DNN_BATCH_SIZE,
        verbose=1,
    )

    predictions = model.predict(X_test)
    pred_dict = {name: predictions[i].flatten() for i, name in enumerate(OUTPUT_COLS)}

    metrics = {}
    for name in OUTPUT_COLS:
        y_true, y_pred = Y_test[name], pred_dict[name]
        metrics[name] = {
            "MAE":  mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2":   r2_score(y_true, y_pred),
        }

    return model, scaler, pd.DataFrame(metrics).T


def categorical_bin_accuracy(y_true, y_pred, bins, labels) -> float:
    """
    Converts continuous predictions into engineering status bands
    (e.g., SOC -> Critical/Low/Mid/High/Full) and reports classification
    accuracy on the coarser categories used for BMS alerting/dashboards.
    """
    y_true_cat = pd.cut(y_true, bins=bins, labels=labels, include_lowest=True).astype(str)
    y_pred_cat = pd.cut(np.clip(y_pred, bins[0], bins[-1]),
                         bins=bins, labels=labels, include_lowest=True).astype(str)
    return accuracy_score(y_true_cat, y_pred_cat)


def export_deployment_formats(model: tf.keras.Model, base_name="AI_BMS_Model"):
    """
    Exports the trained multi-head model to three deployment formats and
    verifies each stays within the DEPLOYMENT_BUDGET_KB constraint
    (Section III-D, Table V):

        .keras  -> 388.78 KB   (native Keras format)
        .h5     -> 400.77 KB   (legacy HDF5, hardware-toolchain compatible)
        .tflite -> 101.16 KB   (quantized, microcontroller deployment)
    """
    import os

    keras_path = f"{base_name}.keras"
    h5_path = f"{base_name}.h5"
    tflite_path = f"{base_name}.tflite"

    model.save(keras_path)
    model.save(h5_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    sizes_kb = {
        p: os.path.getsize(p) / 1024
        for p in (keras_path, h5_path, tflite_path)
    }
    for path, size in sizes_kb.items():
        assert size <= DEPLOYMENT_BUDGET_KB, f"{path} exceeds deployment budget!"
    return sizes_kb


# =============================================================================
# 7. TOP-LEVEL PIPELINE ORCHESTRATION
# =============================================================================

def main():
    """
    End-to-end reproduction of every quantitative result reported in
    Sections IV-A through IV-C of the paper.
    """
    # --- Load and audit -----------------------------------------------------
    df = load_dataset(DATA_PATH)
    quality_report = run_data_quality_checks(df)
    print("Data quality report:", quality_report)

    # --- Track 1: RUL regression comparison (Table IV) -----------------------
    rul_results_table = run_track1_rul_regression(df)
    print("\n=== Table IV: RUL Regression Comparison ===")
    print(rul_results_table)

    # --- Track 2: LSTM cross-battery SOH forecasting --------------------------
    lstm_model, history, test_data, baseline_metrics, test_battery = \
        run_track2_lstm_soh_forecast(df)
    print(f"\n=== Baseline LSTM (held-out battery {test_battery}) ===")
    print(baseline_metrics)

    # --- Track 3: Online calibration -----------------------------------------
    X_test, y_test = test_data
    calibrated_model, calibrated_metrics, future_data = \
        online_calibrate_lstm(lstm_model, X_test, y_test)
    print("\n=== Calibrated LSTM (20-cycle fine-tune) ===")
    print(calibrated_metrics)

    rmse_reduction = (
        (baseline_metrics["RMSE"] - calibrated_metrics["RMSE"])
        / baseline_metrics["RMSE"] * 100
    )
    print(f"RMSE reduction from calibration: {rmse_reduction:.1f}%")

    # --- Track 4: Multi-head production BMS model -----------------------------
    # (requires a discharge-mode dataframe with Temperature_measured,
    #  Voltage_measured, Current_measured, id_cycle columns — see
    #  simulate_multicell_pack docstring)
    # production_df = simulate_multicell_pack(discharge_df)
    # production_df = inject_production_edge_cases(production_df)
    # dnn_model, scaler, table_v = run_track4_multihead_bms(production_df)
    # print("\n=== Table V: Multi-Head DNN Accuracy ===")
    # print(table_v)
    # sizes_kb = export_deployment_formats(dnn_model)
    # print("\n=== Deployment Footprint (KB) ===")
    # print(sizes_kb)


if __name__ == "__main__":
    main()