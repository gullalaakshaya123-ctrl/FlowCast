import pandas as pd

# Load traffic + weather merged data
traffic_weather = pd.read_csv(
    "data/interim/traffic_weather_merged.csv"
)

# Load cleaned calendar data
calendar = pd.read_csv(
    "data/interim/calendar_dates.csv"
)

# Use traffic date as the main date
traffic_weather["date"] = pd.to_datetime(
    traffic_weather["date_x"]
)

calendar["date"] = pd.to_datetime(
    calendar["date"]
)

# Merge calendar information using date
merged = pd.merge(
    traffic_weather,
    calendar,
    on="date",
    how="left"
)

print("========== CALENDAR MERGE RESULT ==========")
print("Traffic + Weather records:", len(traffic_weather))
print("Final merged records:", len(merged))

print("\n========== CALENDAR MATCHING ==========")

print(
    "Missing public_holiday:",
    merged["public_holiday"].isna().sum()
)

print(
    "Missing holiday_name:",
    merged["holiday_name"].isna().sum()
)

print(
    "Missing event_flag:",
    merged["event_flag"].isna().sum()
)

print(
    "Missing event_name:",
    merged["event_name"].isna().sum()
)

print(
    "Missing roadwork_flag:",
    merged["roadwork_flag"].isna().sum()
)

print("\n========== SAMPLE FINAL DATA ==========")

print(
    merged[
        [
            "road_id",
            "date",
            "public_holiday",
            "holiday_name",
            "event_flag",
            "event_name",
            "roadwork_flag"
        ]
    ].head(10)
)

# Save final integrated dataset
merged.to_csv(
    "data/interim/flowcast_integrated.csv",
    index=False
)

print("\nTraffic + Weather + Calendar merge completed successfully.")
print("Integrated dataset saved successfully.")