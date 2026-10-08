# FlowCast 🚦

## AI-Powered Smart Traffic Flow Prediction Platform

FlowCast is an AI-powered traffic analytics and prediction platform that combines traffic sensor data, weather information, and calendar events to analyze traffic conditions, classify congestion levels, and forecast short-term traffic volume.

The project combines classical machine learning and deep learning to provide an end-to-end traffic intelligence pipeline.

---

## 🚀 Key Features

- Traffic data validation and cleaning
- Duplicate and invalid-value handling
- Missing-value imputation
- Weather data cleaning and integration
- Calendar and event data integration
- Date-time standardization
- Traffic feature engineering
- Exploratory Data Analysis (EDA)
- Congestion-level classification using Machine Learning
- Traffic-volume forecasting using LSTM
- Interactive Streamlit dashboard
- Model performance comparison
- Time-based model evaluation

---

## 🎯 Project Objectives

- Analyze historical traffic patterns
- Identify different levels of traffic congestion
- Study the influence of weather and calendar conditions
- Classify traffic congestion using machine learning
- Forecast the next 30-minute traffic volume
- Provide an interactive dashboard for traffic monitoring and prediction

---

## 📊 Dataset

FlowCast uses three datasets.

### Traffic Sensor Data

Contains:

- Road ID and road name
- Latitude and longitude
- Date and time
- Traffic volume
- Vehicle count
- Vehicle type distribution
- Average speed
- Occupancy
- Congestion level
- Travel time
- Accident count
- Signal timing
- Road capacity

### Weather Data

Contains:

- Weather station
- Date and time
- Weather condition
- Temperature
- Rainfall
- Visibility

### Calendar Data

Contains:

- Public holiday information
- Holiday names
- Event information
- Event names
- Roadwork information

---

## 🔄 Data Processing Pipeline

```text
Raw Traffic Data
       |
       v
Data Validation
       |
       v
Duplicate Removal
       |
       v
Invalid Value Handling
       |
       v
Missing Value Imputation
       |
       v
Weather Cleaning
       |
       v
Calendar Cleaning
       |
       v
Date-Time Standardization
       |
       v
Traffic + Weather + Calendar Integration
       |
       v
Feature Engineering
       |
       v
EDA & Data Quality Analysis
       |
       +----------------------+
       |                      |
       v                      v
 Classical ML           Deep Learning
       |                      |
       v                      v
    XGBoost                 LSTM
       |                      |
       +----------+-----------+
                  |
                  v
        Streamlit Dashboard