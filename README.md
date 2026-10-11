# FlowCast: AI-Powered Smart Traffic Flow Prediction Platform

FlowCast is an AI-powered traffic analysis and prediction platform designed to analyze traffic conditions, classify congestion levels, and forecast future traffic volume using machine learning and deep learning.

The project combines **XGBoost for congestion prediction**, **LSTM for traffic volume forecasting**, and an interactive **Streamlit dashboard** to present traffic insights in an easy-to-understand format.

## Project Overview

* **Project Name:** FlowCast
* **Domain:** Artificial Intelligence and Machine Learning
* **Language:** Python
* **Dashboard:** Streamlit
* **Machine Learning:** Random Forest, XGBoost
* **Deep Learning:** Long Short-Term Memory (LSTM)
* **Repository:** [FlowCast on GitHub](https://github.com/gullalaakshaya123-ctrl/FlowCast)

## Project Architecture

![FlowCast Project Architecture](reports/figures/flowcast_architecture.png)

FlowCast follows an end-to-end machine learning workflow:

* **Data Collection:** Traffic sensor logs, weather observations, and calendar events.
* **Data Preprocessing:** Duplicate removal, invalid-value handling, missing-value treatment, and date standardization.
* **Feature Engineering:** Creation of time-based, weather-related, vehicle-ratio, and traffic-capacity features.
* **Model Development:** XGBoost classifies congestion, while LSTM forecasts traffic volume.
* **Evaluation:** Models are evaluated using suitable classification and regression metrics.
* **Dashboard:** Streamlit presents predictions, forecasts, charts, and traffic records.

## Streamlit Dashboard

The interactive dashboard brings traffic analysis and predictions together in one interface.

![FlowCast Dashboard](reports/screenshots/dashboard_overview.png)

Dashboard features include:

* Traffic volume and average-speed analysis
* Road selection and traffic trend visualization
* Congestion classification using XGBoost
* Traffic volume forecasting using LSTM
* Weather and congestion visualizations
* Model performance metrics
* Recent traffic records

## Models and Performance

### XGBoost Congestion Classification

XGBoost predicts traffic congestion categories using traffic, road, time, weather, and vehicle-related features.

| Metric            | Validation Result |
| ----------------- | ----------------: |
| Accuracy          |            99.27% |
| Macro F1-score    |            98.60% |
| Weighted F1-score |            99.27% |

The model was evaluated using a time-based split of records with originally available congestion labels. These results should be interpreted in the context of the dataset and validation procedure.

### Random Forest Congestion Classification

Random Forest was evaluated as a comparison model.

| Metric            | Validation Result |
| ----------------- | ----------------: |
| Accuracy          |            98.36% |
| Macro F1-score    |            97.09% |
| Weighted F1-score |            98.37% |

### LSTM Traffic Volume Forecasting

The LSTM model uses historical traffic observations to forecast traffic volume for the next 30-minute interval.

| Metric   | Validation Result |
| -------- | ----------------: |
| MAE      |             54.99 |
| RMSE     |             85.47 |
| R² score |            0.9062 |

The LSTM model uses a sequence of 12 historical observations, representing approximately six hours of data when observations are 30 minutes apart.

**Note:** Classification metrics and regression metrics measure different tasks and should not be compared directly.

## Dataset

The project uses three input datasets:

* `traffic_sensor_log.csv` — traffic sensor measurements and congestion information.
* `weather_observations.csv` — weather conditions, temperature, rainfall, and visibility.
* `calendar_events.csv` — holidays, events, and roadwork information.

The traffic dataset originally contained 178,468 records. After duplicate removal and cleaning, 176,701 traffic records remained.

The dataset covers January 1 to May 31, 2025, across 25 roads.

## Data Preprocessing and Feature Engineering

The preprocessing pipeline includes:

* Removing duplicate traffic records.
* Identifying invalid traffic volume, speed, and occupancy values.
* Handling missing numeric and weather values.
* Standardizing dates and weather-condition labels.
* Integrating traffic, weather, and calendar data.
* Creating time-based features such as hour, day of week, weekends, and rush-hour indicators.
* Creating traffic-capacity ratios and vehicle-type ratios.
* Creating weather-related indicators, including rain, fog, and poor visibility.
* Separating originally available congestion labels from model-imputed labels for evaluation.

### Data Quality Note

Some missing congestion labels were reconstructed using a machine learning model. These reconstructed labels are retained for analysis, but they are **not treated as ground-truth labels** when reporting the primary classification validation metrics.

## Technology Stack

* **Python** — programming and model development
* **Pandas and NumPy** — data manipulation and numerical operations
* **Scikit-learn** — preprocessing, Random Forest, evaluation, and metrics
* **XGBoost** — congestion classification
* **TensorFlow / Keras** — LSTM forecasting
* **Matplotlib and Seaborn** — data visualization
* **Streamlit** — interactive dashboard
* **Joblib** — saving and loading compatible model artifacts

## Project Structure

```text
FlowCast/
├── dashboards/
│   └── app.py
├── data/
│   └── raw/
│       ├── traffic_sensor_log.csv
│       ├── weather_observations.csv
│       └── calendar_events.csv
├── models/
│   └── # Locally saved model artifacts
├── reports/
│   ├── figures/
│   │   └── flowcast_architecture.png
│   ├── screenshots/
│   │   └── dashboard_overview.png
│   ├── classical_model_comparison.csv
│   ├── model_comparison.csv
│   └── FLOWCAST_MODEL_CARD.md
├── src/
│   ├── data_validation.py
│   ├── feature_engineering.py
│   ├── prepare_ml_dataset.py
│   ├── xgboost_congestion_model.py
│   ├── train_lstm_model.py
│   └── # Additional preprocessing and evaluation scripts
├── .gitignore
├── README.md
└── requirements.txt
```

*Note: Large model files and generated intermediate datasets are excluded from Git through `.gitignore`. The complete set of source scripts is available in the repository.*

## Installation and Setup

### Prerequisites

* Python 3.10 or a compatible Python version for the installed dependencies
* Git
* Visual Studio Code or another Python editor

### 1. Clone the repository

```bash
git clone https://github.com/gullalaakshaya123-ctrl/FlowCast.git
cd FlowCast
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Run the dashboard

```powershell
streamlit run dashboards/app.py
```

Streamlit will provide a local URL to open the dashboard in your browser.

**Important:** Running the dashboard requires the necessary dataset and trained model artifacts to be available at the paths expected by the application. These artifacts are not all included in the Git repository, so a fresh clone may require running the relevant preprocessing and training scripts first.

## Evaluation Approach

The congestion models use a time-based train-validation split to evaluate performance on later records than those used for training.

The LSTM forecasting model is evaluated separately using regression metrics. Its feature scaler is fitted on training data to avoid using validation data during scaler fitting.

Results represent performance on the available dataset and should not be interpreted as a guarantee of performance on every real-world road or traffic condition.

## Future Improvements

* Add live traffic sensor data integration.
* Deploy the dashboard to a public hosting platform.
* Add automated data updates and model retraining.
* Evaluate the models on additional roads and longer time periods.
* Improve forecasting with additional external factors and more recent observations.
* Add monitoring for data quality and prediction drift.

## Learning Outcomes

Through FlowCast, I worked on:

* Data cleaning and exploratory data analysis
* Feature engineering and dataset integration
* Machine learning model training and evaluation
* Time-series sequence preparation and LSTM forecasting
* Interactive dashboard development
* Model documentation and GitHub version control

## Author

**Gullala Akshaya**

B.Tech — Artificial Intelligence and Machine Learning

[GitHub Repository](https://github.com/gullalaakshaya123-ctrl/FlowCast)

---

*FlowCast — Better Data, Smarter Models, and More Informed Traffic Decisions.*
## 🐳 Run FlowCast with Docker

FlowCast can run inside a Docker container using Streamlit.

### Prerequisites
- Docker Desktop installed and running.

### Build the Docker Image

```bash
docker build -t flowcast-app .
```

### Run the Application

```bash
docker run --rm -p 8501:8501 flowcast-app
```

### Open the Dashboard

Open your browser and visit:

http://localhost:8501

To stop the application, press `Ctrl + C` in the terminal running Docker.
