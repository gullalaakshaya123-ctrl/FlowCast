import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FlowCast - Smart Traffic Prediction",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🚦 FlowCast")
st.subheader("AI-Powered Smart Traffic Flow Prediction Platform")

st.write(
    "FlowCast analyzes traffic, weather and calendar information "
    "to understand congestion and forecast future traffic volume."
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        "data/interim/flowcast_features.csv"
    )

    data["datetime"] = pd.to_datetime(
        data["datetime"]
    )

    return data


@st.cache_resource
def load_models():

    xgb_model = joblib.load(
        "models/xgboost_congestion_model.joblib"
    )

    lstm_model = load_model(
        "models/flowcast_lstm_best.keras"
    )

    feature_scaler = joblib.load(
        "models/lstm_feature_scaler.joblib"
    )

    target_scaler = joblib.load(
        "models/lstm_target_scaler.joblib"
    )

    return (
        xgb_model,
        lstm_model,
        feature_scaler,
        target_scaler
    )


data = load_data()

(
    xgb_model,
    lstm_model,
    feature_scaler,
    target_scaler
) = load_models()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Dashboard Controls")

roads = sorted(
    data["road_name"].dropna().unique()
)

selected_road = st.sidebar.selectbox(
    "Select Road",
    roads
)


# ============================================================
# FILTER DATA
# ============================================================

road_data = data[
    data["road_name"] == selected_road
].copy()

road_data = road_data.sort_values(
    "datetime"
)


# ============================================================
# OVERVIEW METRICS
# ============================================================

st.header("📊 Traffic Overview")

col1, col2, col3, col4 = st.columns(4)

latest_row = road_data.iloc[-1]

col1.metric(
    "Traffic Volume",
    f"{latest_row['traffic_volume']:.0f}"
)

col2.metric(
    "Average Speed",
    f"{latest_row['avg_speed']:.1f}"
)

col3.metric(
    "Occupancy",
    f"{latest_row['occupancy']:.1f}%"
)

col4.metric(
    "Congestion Level",
    latest_row["congestion_level"]
)


# ============================================================
# TRAFFIC TREND
# ============================================================

st.header("📈 Traffic Volume Trend")

chart_data = road_data[
    ["datetime", "traffic_volume"]
].set_index("datetime")

st.line_chart(
    chart_data
)


# ============================================================
# CONGESTION DISTRIBUTION
# ============================================================

st.header("🚦 Congestion Distribution")

congestion_counts = (
    data["congestion_level"]
    .value_counts()
    .rename_axis("Congestion Level")
    .reset_index(name="Count")
)

st.bar_chart(
    congestion_counts.set_index(
        "Congestion Level"
    )
)


# ============================================================
# WEATHER DISTRIBUTION
# ============================================================

st.header("🌦️ Weather Distribution")

weather_counts = (
    data["weather_condition"]
    .value_counts()
    .rename_axis("Weather")
    .reset_index(name="Count")
)

st.bar_chart(
    weather_counts.set_index(
        "Weather"
    )
)


# ============================================================
# XGBOOST CONGESTION PREDICTION
# ============================================================

st.header("🤖 XGBoost Congestion Prediction")

st.write(
    "The trained XGBoost model predicts the congestion "
    "level using current traffic and environmental features."
)


xgb_features = [
    "traffic_volume",
    "avg_speed",
    "occupancy",
    "road_capacity",
    "vehicle_count",
    "accident_count",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_rush_hour",
    "is_rain",
    "is_fog",
    "is_poor_visibility",
    "vehicle_2w_ratio",
    "vehicle_car_ratio",
    "vehicle_lcv_ratio",
    "vehicle_hcv_ratio"
]


prediction_input = pd.DataFrame(
    [latest_row[xgb_features].values],
    columns=xgb_features
)

xgb_prediction = xgb_model.predict(
    prediction_input
)[0]


label_mapping = {
    0: "Free-flow",
    1: "Moderate",
    2: "Heavy",
    3: "Severe"
}

predicted_congestion = label_mapping.get(
    int(xgb_prediction),
    str(xgb_prediction)
)

st.success(
    f"Predicted Congestion Level: **{predicted_congestion}**"
)


# ============================================================
# LSTM TRAFFIC FORECAST
# ============================================================

st.header("🔮 LSTM Traffic Volume Forecast")

st.write(
    "The LSTM uses the previous 6 hours of traffic information "
    "to forecast the next 30-minute traffic volume."
)


lstm_features = [
    "traffic_volume",
    "avg_speed",
    "occupancy",
    "vehicle_count",
    "road_capacity"
]


# Use the latest 12 observations
latest_sequence = road_data[
    lstm_features
].tail(12)


if len(latest_sequence) == 12:

    sequence = latest_sequence.values

    sequence_scaled = feature_scaler.transform(
        sequence
    )

    sequence_scaled = sequence_scaled.reshape(
        1,
        12,
        5
    )

    forecast_scaled = lstm_model.predict(
        sequence_scaled,
        verbose=0
    )

    forecast = target_scaler.inverse_transform(
        forecast_scaled
    )[0][0]

    st.metric(
        "Forecasted Traffic Volume (Next 30 min)",
        f"{forecast:.0f}"
    )

else:

    st.warning(
        "Not enough historical observations available "
        "for LSTM forecasting."
    )


# ============================================================
# LATEST DATA
# ============================================================

st.header("📋 Latest Traffic Records")

display_columns = [
    "datetime",
    "road_name",
    "traffic_volume",
    "avg_speed",
    "occupancy",
    "weather_condition",
    "congestion_level"
]

st.dataframe(
    road_data[
        display_columns
    ].tail(20),
    use_container_width=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "FlowCast | AI-Powered Smart Traffic Flow Prediction Platform"
)