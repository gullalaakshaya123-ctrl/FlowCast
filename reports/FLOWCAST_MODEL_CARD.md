# FlowCast Model Card

## Project

**Project Name:** FlowCast – AI-Powered Smart Traffic Flow Prediction Platform

**Project ID:** PROJ-FLOW-260711

**Domain:** Artificial Intelligence

**Objective:**  
FlowCast is an AI-based traffic analytics and prediction platform designed to analyze traffic conditions, predict congestion levels, and forecast future traffic volume using machine learning and deep learning techniques.

---

## Dataset

The project uses three main datasets:

- Traffic sensor data
- Weather observations
- Calendar and event information

### Traffic Dataset

- Original records: 178,468
- Records after duplicate removal: 176,701
- Number of roads: 25
- Date range: January 1, 2025 to May 31, 2025
- Sampling interval: approximately 30 minutes

### Weather Dataset

- Records: 10,872
- Date range: January 1, 2025 to May 31, 2025
- Weather conditions include Clear, Cloudy, Rain, Overcast and Fog

### Calendar Dataset

- Records: 151
- Contains public holidays, events and roadwork information

---

## Data Preprocessing

The following preprocessing operations were performed:

- Duplicate traffic records were identified and removed.
- Invalid negative traffic-volume values were replaced with missing values.
- Impossible average-speed values were replaced with missing values.
- Occupancy values greater than 100% were treated as invalid.
- Missing numerical traffic values were filled using median imputation.
- Weather condition labels were standardized.
- Missing weather temperature and visibility values were filled using median imputation.
- Calendar event and holiday names were completed for non-event and non-holiday records.
- Date and time formats were standardized.
- Traffic, weather and calendar datasets were integrated.
- Time-based features were created.
- Traffic-volume and capacity-related features were created.
- Weather-related indicator features were created.
- Vehicle-type distribution was converted into numerical ratio features.

---

## Feature Engineering

Important engineered features include:

- Hour
- Day of week
- Weekend indicator
- Month
- Day
- Rush-hour indicator
- Volume-to-capacity ratio
- Speed-to-capacity ratio
- Vehicle-to-capacity ratio
- Accident indicator
- Rain indicator
- Fog indicator
- Poor-visibility indicator
- Two-wheeler ratio
- Car ratio
- LCV ratio
- HCV ratio

---

# Model 1: Random Forest

## Task

**Congestion-level classification**

The Random Forest model predicts one of four congestion classes:

- Free-flow
- Moderate
- Heavy
- Severe

## Validation Strategy

A time-based validation strategy was used.

Training data:

- January 1, 2025 to March 31, 2025

Validation data:

- April 1, 2025 to May 31, 2025

This prevents randomly mixing future observations into the training data.

## Performance

- Accuracy: **98.36%**
- Macro F1: **97.09%**
- Weighted F1: **98.37%**

---

# Model 2: XGBoost

## Task

**Congestion-level classification**

The XGBoost model predicts:

- Free-flow
- Moderate
- Heavy
- Severe

## Validation Strategy

The same time-based training and validation split was used.

## Performance

- Accuracy: **99.27%**
- Macro F1: **98.60%**
- Weighted F1: **99.27%**

## Result

XGBoost achieved the strongest classification performance among the classical machine learning models evaluated.

---

# Model 3: LSTM

## Task

**Traffic-volume forecasting**

Unlike the Random Forest and XGBoost models, the LSTM is a regression model.

It predicts the next traffic-volume value using historical traffic observations.

## Input

The LSTM uses:

- Traffic volume
- Average speed
- Occupancy
- Vehicle count
- Road capacity

The model uses 12 historical observations.

Since observations are approximately 30 minutes apart:

**12 observations = approximately 6 hours of historical traffic information.**

## Architecture

The model contains:

- LSTM layer with 64 units
- Dropout layer
- LSTM layer with 32 units
- Dropout layer
- Dense layer with 16 units
- Output layer with 1 unit

## Training

- Optimizer: Adam
- Loss function: Mean Squared Error
- Batch size: 128
- Maximum epochs: 30
- Early stopping: Enabled

Training stopped after 15 epochs because validation performance stopped improving.

## Validation Performance

- MAE: **54.99**
- RMSE: **85.47**
- R²: **0.9062**

The R² value indicates that the model explains approximately 90.62% of the variation in validation traffic volume.

---

# Model Comparison

| Model | Task | Main Metric | Result |
|---|---|---:|---:|
| Random Forest | Congestion Classification | Accuracy | 98.36% |
| XGBoost | Congestion Classification | Accuracy | 99.27% |
| LSTM | Traffic Volume Forecasting | R² | 0.9062 |

The classification models and LSTM solve different prediction problems.

Therefore, classification accuracy should not be directly compared with LSTM regression metrics.

XGBoost is currently the preferred classical model for congestion classification, while LSTM provides the traffic-volume forecasting capability.

---

# Model Outputs

The trained models are stored in the `models` directory.

### Random Forest

`models/baseline_congestion_model.joblib`

### XGBoost

`models/xgboost_congestion_model.joblib`

### LSTM

`models/flowcast_lstm_best.keras`

`models/flowcast_lstm_final.keras`

### LSTM Scalers

`models/lstm_feature_scaler.joblib`

`models/lstm_target_scaler.joblib`

---

# Data Quality and Label Considerations

A portion of the original traffic dataset contained missing congestion labels.

The missing congestion labels were reconstructed using a machine-learning-based approach so that the integrated dataset could support downstream analysis.

For reliable model evaluation, a separate evaluation dataset containing only originally observed congestion labels was created.

Original observed congestion labels used for evaluation:

- 150,077 records

Model-imputed congestion labels:

- 26,624 records

The classification evaluation results reported above are based on the originally observed labels rather than the reconstructed labels.

---

# Limitations

- The traffic dataset covers approximately five months.
- Traffic patterns may change over longer periods.
- The LSTM currently uses five numerical traffic features and does not explicitly encode road identity.
- Missing congestion labels required model-based reconstruction for the integrated dataset.
- Weather and calendar information may have limited influence for some traffic observations.
- Real-world deployment would require continuous monitoring and periodic model retraining.
- Predictions may be less reliable during unusual events that are not represented sufficiently in the training data.

---

# Responsible Use

FlowCast is intended to support traffic analysis and prediction.

Predictions should be treated as decision-support information rather than guaranteed future outcomes.

Real-world deployment should include continuous monitoring of model performance, data quality and prediction errors.

---

# Current Best Models

**Best classical classification model:** XGBoost

**Best forecasting model:** LSTM

**XGBoost validation accuracy:** 99.27%

**LSTM validation R²:** 0.9062

---

## Project Status

- M1 Ingestion & Validation: Completed
- M2 Cleaning & Merging: Completed
- M3 Feature Engineering: Completed
- M4 EDA & Data Quality: Completed
- M5 Classical ML: Completed
- M6 Deep Learning: Completed
- M7 Streamlit Dashboard: In Progress