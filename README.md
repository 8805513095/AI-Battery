AI-Driven Battery SOH, RUL, and Multi-Parameter BMS Prediction

This repository contains the implementation and reproducibility workflow for the research paper:

“AI-Driven State-of-Health, Remaining Useful Life, and Multi-Parameter Battery Management: A Comparative Machine Learning and Deep Learning Study on the NASA Li-Ion Battery Data.”

Author: Vinothkumar Kolluru

The project presents an end-to-end machine learning and deep learning framework for lithium-ion battery health monitoring, including:

Remaining Useful Life prediction using ensemble regression
Sequential State-of-Health forecasting using LSTM
Online calibration for previously unseen batteries
Multi-output Battery Management System estimation
TensorFlow Lite model conversion for embedded deployment
Dataset

The experiments are based on the NASA Li-Ion Battery Aging Datasets published through the NASA Open Data Portal.

Dataset source:

https://data.nasa.gov/dataset/li-ion-battery-aging-datasets

The original NASA dataset contains lithium-ion battery charge, discharge, and impedance measurements collected under different operating and temperature conditions.

This project uses a cleaned tabular version derived from the NASA dataset. The processed dataset used in the study contains:

Property	Value
Total records	7,368
Number of batteries	34
Ambient temperatures	4°C, 22°C, 24°C, 43°C, and 44°C
Dataset columns	9
Missing values	0
Zero-capacity artifacts	50 records

The expected cleaned dataset fields are:

Column	Description
type	Cycle-operation indicator
ambient_temperature	Battery operating temperature
battery_id	Unique battery identifier
test_id	Sequential test identifier
uid	Unique record identifier
filename	Original NASA measurement-file reference
Capacity	Measured discharge capacity
Re	Electrolyte resistance
Rct	Charge-transfer resistance

The cycle-operation indicator is encoded as:

-1 = discharge
 0 = charge
 1 = impedance measurement

The repository does not need to redistribute the original NASA dataset. Download it directly from the NASA data portal and follow NASA’s applicable data-use and attribution requirements.

Research Objectives

The study contains four experimental tracks.

Track 1: Remaining Useful Life Regression

Four regression algorithms are compared:

Random Forest
XGBoost
Gradient Boosting
Support Vector Regression

A physics-motivated degradation feature is constructed from electrolyte and charge-transfer resistance:

degradation_feature = Re × Rct

The resistance-based RUL proxy used in the paper is:

RUL = 1000 / (degradation_feature + 1)

The feature matrix includes the available numerical battery measurements after excluding the target, intermediate degradation feature, and non-numerical filename field.

Track 2: Sequential SOH Forecasting

Battery State of Health is defined as:

SOH(t) = Capacity(t) / Capacity(0)

A sliding window containing ten consecutive SOH observations is used to predict the next-cycle SOH value.

The LSTM architecture consists of:

Input sequence: 10 × 1
LSTM: 64 units, return sequences
Dropout: 40%
LSTM: 32 units
Dropout: 40%
Dense: 16 units, ReLU
Output: 1 unit, linear

The model contains approximately 29,857 trainable parameters.

A strict leave-one-battery-out evaluation is used. All cycles from the held-out battery are excluded from baseline model training.

Track 3: Online Calibration

The pretrained LSTM is adapted to a previously unseen battery using only its first 20 available observations.

During calibration:

Both LSTM layers remain frozen
Only the dense prediction layers are updated
The calibration learning rate is (5 \times 10^{-5})
Training runs for up to 20 epochs
Early stopping is applied
A mean-residual bias correction is added to future predictions

This procedure represents a realistic BMS commissioning scenario in which only limited measurements from a newly installed battery are initially available.

Track 4: Multi-Parameter BMS Estimation

A seven-output neural network jointly predicts:

State of Charge
State of Health
Rate of discharge
Pack battery level
Remaining battery life
Cycle life
Depth of discharge

The eight BMS input channels are:

Cell 1 temperature
Cell 2 temperature
Cell 3 temperature
Cell 4 temperature
Total pack voltage
Total pack current
Total pressure
Gas-sensor signal

The multi-head architecture uses:

Input: 8 sensor features
Shared Dense: 128 units, ReLU
Batch Normalization
Dropout: 20%
Shared Dense: 64 units, ReLU

Seven output branches:
    Dense: 32 units, ReLU
    Output: 1 unit, linear
Repository Structure
battery-soh-rul-bms/
│
├── README.md
├── requirements.txt
├── battery_soh_rul_bms_journal_pseudocode.py
│
├── data/
│   └── NASA_Battery_Dataset_Cleaned.csv
│
├── journal_outputs/
│   ├── figures/
│   ├── tables/
│   ├── models/
│   ├── predictions/
│   ├── reports/
│   └── intermediate/
│
└── paper/
    └── manuscript.tex

The dataset filename can be changed through the command-line arguments.

Installation

Clone the repository:

git clone https://github.com/USERNAME/battery-soh-rul-bms.git
cd battery-soh-rul-bms

Create a virtual environment:

python -m venv .venv

Activate it on macOS or Linux:

source .venv/bin/activate

Activate it on Windows:

.venv\Scripts\activate

Install the required packages:

pip install numpy pandas scipy scikit-learn matplotlib joblib xgboost tensorflow

A corresponding requirements.txt may contain:

numpy
pandas
scipy
scikit-learn
matplotlib
joblib
xgboost
tensorflow
Data Preparation

Download the NASA Li-Ion Battery Aging Dataset:

https://data.nasa.gov/dataset/li-ion-battery-aging-datasets

Convert or clean the original NASA files into a tabular CSV containing the required columns.
Save the processed dataset as:
data/NASA_Battery_Dataset_Cleaned.csv
Confirm that numerical fields are correctly parsed and that battery records are chronologically ordered using:
battery_id
test_id
uid

Zero-capacity records should be removed or carefully interpolated before SOH normalization.

Running the Complete Pipeline
python battery_soh_rul_bms_journal_pseudocode.py \
    --csv-path data/NASA_Battery_Dataset_Cleaned.csv \
    --output-dir journal_outputs

This runs all four experimental tracks.

Running Individual Experiments
RUL Regression Only
python battery_soh_rul_bms_journal_pseudocode.py \
    --csv-path data/NASA_Battery_Dataset_Cleaned.csv \
    --output-dir journal_outputs \
    --tracks track1_rul
LSTM SOH Forecasting
python battery_soh_rul_bms_journal_pseudocode.py \
    --csv-path data/NASA_Battery_Dataset_Cleaned.csv \
    --output-dir journal_outputs \
    --tracks track2_lstm \
    --held-out-battery 56 \
    --sequence-length 10
LSTM with Online Calibration
python battery_soh_rul_bms_journal_pseudocode.py \
    --csv-path data/NASA_Battery_Dataset_Cleaned.csv \
    --output-dir journal_outputs \
    --tracks track2_lstm track3_calibration \
    --held-out-battery 56 \
    --calibration-observations 20
Multi-Parameter BMS Model
python battery_soh_rul_bms_journal_pseudocode.py \
    --csv-path data/NASA_Battery_Dataset_Cleaned.csv \
    --output-dir journal_outputs \
    --tracks track4_multihead
RUL Experiment Modes

Two RUL modes are available.

Paper-Reproduction Mode
--rul-mode paper_reproduction

This mode retains Re and Rct as input features, matching the methodology reported in the paper.

python battery_soh_rul_bms_journal_pseudocode.py \
    --csv-path data/NASA_Battery_Dataset_Cleaned.csv \
    --tracks track1_rul \
    --rul-mode paper_reproduction
Leakage-Safe Robustness Mode
--rul-mode leakage_safe

This mode removes Re, Rct, and the directly derived degradation feature from the predictor matrix.

python battery_soh_rul_bms_journal_pseudocode.py \
    --csv-path data/NASA_Battery_Dataset_Cleaned.csv \
    --tracks track1_rul \
    --rul-mode leakage_safe

This experiment is recommended as an additional robustness analysis because the paper’s RUL proxy is mathematically derived from Re × Rct.

Reported Results
RUL Regression
Model	MSE	MAE	R²
Random Forest	0.00286	0.0181	0.99987
XGBoost	0.03133	0.1183	0.99863
Gradient Boosting	Intermediate	Intermediate	Intermediate
Support Vector Regression	Highest	Highest	Lowest

Random Forest achieved the strongest reported RUL-regression performance.

Cross-Battery SOH Forecasting
Model	RMSE	MAE
Baseline LSTM	0.1221	0.1202
Calibrated LSTM	0.0271	0.0209

The proposed online-calibration procedure reduced:

RMSE by approximately 77.8%
MAE by approximately 82.6%
Multi-Head BMS Estimation
Output	R²	MAE	RMSE
SOC	0.973	0.0308	0.0326
SOH	0.913	0.0245	0.0302
Rate of discharge	0.395	0.0126	0.0372
Pack battery level	0.995	1.244	1.355
Battery life	0.998	1.293	1.893
Cycle life	0.999	1.494	1.675
Depth of discharge	0.994	0.0138	0.0149

Five of the seven outputs achieved (R^2>0.97).

Deployment Footprint
Format	Model Size
Keras	388.78 KB
HDF5	400.77 KB
TensorFlow Lite	101.16 KB

TensorFlow Lite conversion reduced the model footprint by approximately 74% compared with the native Keras model.

Generated Outputs

The program automatically creates:

journal_outputs/
├── figures/
├── tables/
├── models/
├── predictions/
├── reports/
└── intermediate/

Typical outputs include:

RUL model-comparison tables
Actual-versus-predicted plots
Residual plots
Feature-importance plots
LSTM learning curves
SOH trajectory plots
Baseline-versus-calibrated comparisons
Multi-head output metrics
Operational-category confusion matrices
Keras, HDF5, and TensorFlow Lite models
Model-size and latency reports
Experiment configuration files
Data-quality reports
Reproducibility

The implementation applies fixed random seeds to:

Python
NumPy
TensorFlow
Scikit-learn estimators
Synthetic BMS data generation

Exact results can still vary because of:

TensorFlow version
CPU or GPU architecture
Numerical precision
Dataset-cleaning choices
Train-test splitting
Available battery records
TensorFlow Lite conversion behavior

For publication-quality reproduction, record the complete software environment:

pip freeze > environment.txt
Important Scientific Limitations
Resistance-Based RUL Target

The RUL value used in the regression track is a physically motivated proxy:

RUL = 1000 / (Re × Rct + 1)

It is not a directly measured ground-truth failure time.

Because Re and Rct are used to generate the target, retaining them as input variables can produce near-perfect performance by reconstructing the target function. Results should therefore be reported together with leakage-safe or battery-grouped robustness experiments.

Synthetic BMS Channels

The four-cell temperature channels, pressure signal, gas-sensor signal, some voltage/current values, and selected output targets are partly simulated or derived from the NASA measurements.

Consequently, the multi-head results demonstrate architectural and deployment feasibility. They should not be interpreted as independently validated field accuracy for a production automotive or grid-storage BMS.

Cross-Battery Generalization

Random row-level splitting can place records from the same battery in both training and test sets. Battery-grouped validation or leave-one-battery-out evaluation should be preferred when making generalization claims.

Future Work

Planned extensions include:

Ground-truth RUL calculation from measured EOL cycles
Validation on independently measured multi-cell battery-pack telemetry
Evaluation across additional battery chemistries
Physics-informed neural-network losses
Uncertainty quantification
Federated battery-learning frameworks
Improved rate-of-discharge prediction
Cross-temperature domain adaptation
Online edge-device learning
Microcontroller latency and energy-consumption benchmarking
