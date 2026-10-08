import pandas as pd
import json


# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(
    "data/interim/flowcast_integrated.csv"
)

# Convert datetime column
df["datetime"] = pd.to_datetime(df["datetime"])


# ==================================================
# TIME FEATURES
# ==================================================

# Hour of the day: 0 to 23
df["hour"] = df["datetime"].dt.hour

# Day of week: Monday = 0, Sunday = 6
df["day_of_week"] = df["datetime"].dt.dayofweek

# Weekend indicator
# Saturday = 5, Sunday = 6
df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

# Month number
df["month"] = df["datetime"].dt.month

# Day of the month
df["day"] = df["datetime"].dt.day

# Rush hour indicator
# Morning: 7 AM to 10 AM
# Evening: 5 PM to 8 PM
df["is_rush_hour"] = (
    ((df["hour"] >= 7) & (df["hour"] <= 10)) |
    ((df["hour"] >= 17) & (df["hour"] <= 20))
).astype(int)


# ==================================================
# TRAFFIC-BASED FEATURES
# ==================================================

# Traffic volume compared with road capacity
df["volume_capacity_ratio"] = (
    df["traffic_volume"] / df["road_capacity"]
)

# Average speed compared with road capacity
df["speed_capacity_ratio"] = (
    df["avg_speed"] / df["road_capacity"]
)

# Vehicle count compared with road capacity
df["vehicle_capacity_ratio"] = (
    df["vehicle_count"] / df["road_capacity"]
)

# Accident indicator
# 1 = accident occurred
# 0 = no accident
df["accident_flag"] = (
    df["accident_count"] > 0
).astype(int)


# ==================================================
# WEATHER-BASED FEATURES
# ==================================================

# Rain indicator
df["is_rain"] = (
    df["weather_condition"] == "Rain"
).astype(int)

# Fog indicator
df["is_fog"] = (
    df["weather_condition"] == "Fog"
).astype(int)

# Poor visibility indicator
# Less than 5000 meters
df["is_poor_visibility"] = (
    df["visibility"] < 5000
).astype(int)

# Extreme temperature indicator
# Below 5°C or above 30°C
df["is_extreme_temperature"] = (
    (df["temperature"] < 5) |
    (df["temperature"] > 30)
).astype(int)


# ==================================================
# VEHICLE TYPE FEATURES
# ==================================================

def extract_vehicle_ratio(value, vehicle_type):
    """
    Extract the proportion of a particular vehicle type
    from the vehicle_type_dist JSON-like column.
    """

    try:
        data = json.loads(value)

        return float(
            data.get(vehicle_type, 0)
        )

    except (TypeError, ValueError, json.JSONDecodeError):
        return 0.0


# Extract vehicle proportions
df["vehicle_2w_ratio"] = df["vehicle_type_dist"].apply(
    lambda x: extract_vehicle_ratio(x, "2W")
)

df["vehicle_car_ratio"] = df["vehicle_type_dist"].apply(
    lambda x: extract_vehicle_ratio(x, "Car")
)

df["vehicle_lcv_ratio"] = df["vehicle_type_dist"].apply(
    lambda x: extract_vehicle_ratio(x, "LCV")
)

df["vehicle_hcv_ratio"] = df["vehicle_type_dist"].apply(
    lambda x: extract_vehicle_ratio(x, "HCV")
)


# ==================================================
# DISPLAY CREATED FEATURES
# ==================================================

print("========== FEATURE ENGINEERING ==========")

print("\nTime features:")
print(
    [
        "hour",
        "day_of_week",
        "is_weekend",
        "month",
        "day",
        "is_rush_hour"
    ]
)

print("\nTraffic features:")
print(
    [
        "volume_capacity_ratio",
        "speed_capacity_ratio",
        "vehicle_capacity_ratio",
        "accident_flag"
    ]
)

print("\nWeather features:")
print(
    [
        "is_rain",
        "is_fog",
        "is_poor_visibility",
        "is_extreme_temperature"
    ]
)

print("\nVehicle type features:")
print(
    [
        "vehicle_2w_ratio",
        "vehicle_car_ratio",
        "vehicle_lcv_ratio",
        "vehicle_hcv_ratio"
    ]
)


# ==================================================
# SAMPLE VEHICLE TYPE DATA
# ==================================================

print("\n========== VEHICLE TYPE SAMPLE ==========")

print(
    df[
        [
            "vehicle_type_dist",
            "vehicle_2w_ratio",
            "vehicle_car_ratio",
            "vehicle_lcv_ratio",
            "vehicle_hcv_ratio"
        ]
    ].head(10)
)


# ==================================================
# FEATURE CHECK
# ==================================================

print("\n========== FEATURE CHECK ==========")

print("Total records:", len(df))

print(
    "Missing hour:",
    df["hour"].isna().sum()
)

print(
    "Missing day_of_week:",
    df["day_of_week"].isna().sum()
)

print(
    "Missing is_weekend:",
    df["is_weekend"].isna().sum()
)

print(
    "Missing month:",
    df["month"].isna().sum()
)

print(
    "Missing day:",
    df["day"].isna().sum()
)

print(
    "Missing is_rush_hour:",
    df["is_rush_hour"].isna().sum()
)

print(
    "Missing volume_capacity_ratio:",
    df["volume_capacity_ratio"].isna().sum()
)

print(
    "Missing speed_capacity_ratio:",
    df["speed_capacity_ratio"].isna().sum()
)

print(
    "Missing vehicle_capacity_ratio:",
    df["vehicle_capacity_ratio"].isna().sum()
)

print(
    "Missing accident_flag:",
    df["accident_flag"].isna().sum()
)

print(
    "Missing is_rain:",
    df["is_rain"].isna().sum()
)

print(
    "Missing is_fog:",
    df["is_fog"].isna().sum()
)

print(
    "Missing is_poor_visibility:",
    df["is_poor_visibility"].isna().sum()
)

print(
    "Missing is_extreme_temperature:",
    df["is_extreme_temperature"].isna().sum()
)

print(
    "Missing vehicle_2w_ratio:",
    df["vehicle_2w_ratio"].isna().sum()
)

print(
    "Missing vehicle_car_ratio:",
    df["vehicle_car_ratio"].isna().sum()
)

print(
    "Missing vehicle_lcv_ratio:",
    df["vehicle_lcv_ratio"].isna().sum()
)

print(
    "Missing vehicle_hcv_ratio:",
    df["vehicle_hcv_ratio"].isna().sum()
)


# ==================================================
# CHECK VEHICLE RATIO SUM
# ==================================================

df["vehicle_ratio_sum"] = (
    df["vehicle_2w_ratio"] +
    df["vehicle_car_ratio"] +
    df["vehicle_lcv_ratio"] +
    df["vehicle_hcv_ratio"]
)

print("\n========== VEHICLE RATIO CHECK ==========")

print(
    "Minimum vehicle ratio sum:",
    df["vehicle_ratio_sum"].min()
)

print(
    "Maximum vehicle ratio sum:",
    df["vehicle_ratio_sum"].max()
)

print(
    "Average vehicle ratio sum:",
    df["vehicle_ratio_sum"].mean()
)


# ==================================================
# REMOVE TEMPORARY CHECK COLUMN
# ==================================================

df.drop(
    columns=["vehicle_ratio_sum"],
    inplace=True
)


# ==================================================
# SAVE FEATURE-ENGINEERED DATASET
# ==================================================

df.to_csv(
    "data/interim/flowcast_features.csv",
    index=False
)

print("\nFeature engineering completed successfully.")
print("Feature-engineered dataset saved successfully.")