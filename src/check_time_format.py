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
# 2. Display sample time values
# --------------------------------------------------

print("========== TRAFFIC TIME VALUES ==========")

print(
    traffic["time"].head(20).to_string(index=False)
)


print("\n========== WEATHER TIME VALUES ==========")

print(
    weather["time"].head(20).to_string(index=False)
)


# --------------------------------------------------
# 3. Count unique time values
# --------------------------------------------------

print("\n========== UNIQUE TIME COUNTS ==========")

print(
    "Traffic unique times:",
    traffic["time"].nunique()
)

print(
    "Weather unique times:",
    weather["time"].nunique()
)


# --------------------------------------------------
# 4. Check missing time values
# --------------------------------------------------

print("\n========== MISSING TIME VALUES ==========")

print(
    "Traffic missing:",
    traffic["time"].isna().sum()
)

print(
    "Weather missing:",
    weather["time"].isna().sum()
)