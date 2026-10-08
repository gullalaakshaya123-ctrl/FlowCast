import pandas as pd


# --------------------------------------------------
# 1. Load standardized datasets
# --------------------------------------------------

traffic = pd.read_csv(
    "data/interim/traffic_dates.csv"
)

weather = pd.read_csv(
    "data/interim/weather_dates.csv"
)


# --------------------------------------------------
# 2. Create combined datetime
# --------------------------------------------------

traffic["datetime"] = pd.to_datetime(
    traffic["date"].astype(str)
    + " "
    + traffic["time"].astype(str)
)


weather["datetime"] = pd.to_datetime(
    weather["date"].astype(str)
    + " "
    + weather["time"].astype(str)
)


# --------------------------------------------------
# 3. Check datetime values
# --------------------------------------------------

print("========== TRAFFIC DATETIME ==========")

print(
    traffic["datetime"].head(10)
)


print("\n========== WEATHER DATETIME ==========")

print(
    weather["datetime"].head(10)
)


# --------------------------------------------------
# 4. Check missing datetime values
# --------------------------------------------------

print("\n========== MISSING DATETIME ==========")

print(
    "Traffic:",
    traffic["datetime"].isna().sum()
)

print(
    "Weather:",
    weather["datetime"].isna().sum()
)


# --------------------------------------------------
# 5. Display datetime ranges
# --------------------------------------------------

print("\n========== DATETIME RANGES ==========")

print(
    "Traffic:",
    traffic["datetime"].min(),
    "to",
    traffic["datetime"].max()
)

print(
    "Weather:",
    weather["datetime"].min(),
    "to",
    weather["datetime"].max()
)


# --------------------------------------------------
# 6. Save prepared datasets
# --------------------------------------------------

traffic.to_csv(
    "data/interim/traffic_datetime.csv",
    index=False
)

weather.to_csv(
    "data/interim/weather_datetime.csv",
    index=False
)


print("\nDatetime preparation completed successfully.")