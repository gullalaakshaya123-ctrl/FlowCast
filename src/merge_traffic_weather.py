import pandas as pd


# --------------------------------------------------
# 1. Load prepared datasets
# --------------------------------------------------

traffic = pd.read_csv(
    "data/interim/traffic_datetime.csv"
)

weather = pd.read_csv(
    "data/interim/weather_datetime.csv"
)


# --------------------------------------------------
# 2. Convert datetime columns
# --------------------------------------------------

traffic["datetime"] = pd.to_datetime(
    traffic["datetime"]
)

weather["datetime"] = pd.to_datetime(
    weather["datetime"]
)


# --------------------------------------------------
# 3. Rename weather station column
# --------------------------------------------------

weather = weather.rename(
    columns={
        "station_id": "weather_station_id"
    }
)


# --------------------------------------------------
# 4. Sort by datetime first
# --------------------------------------------------

traffic = traffic.sort_values(
    ["datetime", "weather_station_id"]
).reset_index(drop=True)

weather = weather.sort_values(
    ["datetime", "weather_station_id"]
).reset_index(drop=True)


# --------------------------------------------------
# 5. Merge using nearest weather observation
# --------------------------------------------------

merged = pd.merge_asof(
    traffic,
    weather,
    on="datetime",
    by="weather_station_id",
    direction="nearest",
    allow_exact_matches=True
)


# --------------------------------------------------
# 6. Check merged dataset
# --------------------------------------------------

print("========== MERGE RESULT ==========")

print(
    "Traffic records:",
    len(traffic)
)

print(
    "Merged records:",
    len(merged)
)


# --------------------------------------------------
# 7. Check weather matching
# --------------------------------------------------

print("\n========== WEATHER MATCHING ==========")

print(
    "Missing weather condition:",
    merged["weather_condition"].isna().sum()
)

print(
    "Missing temperature:",
    merged["temperature"].isna().sum()
)

print(
    "Missing rainfall:",
    merged["rainfall"].isna().sum()
)

print(
    "Missing visibility:",
    merged["visibility"].isna().sum()
)


# --------------------------------------------------
# 8. Display sample
# --------------------------------------------------

print("\n========== SAMPLE MERGED DATA ==========")

print(
    merged[
        [
            "road_id",
            "datetime",
            "weather_station_id",
            "weather_condition",
            "temperature",
            "rainfall",
            "visibility"
        ]
    ].head(10)
)


# --------------------------------------------------
# 9. Save merged dataset
# --------------------------------------------------

merged.to_csv(
    "data/interim/traffic_weather_merged.csv",
    index=False
)


print(
    "\nTraffic + Weather merge completed successfully."
)

print(
    "Merged dataset saved successfully."
)