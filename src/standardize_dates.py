import pandas as pd


# --------------------------------------------------
# 1. Load cleaned datasets
# --------------------------------------------------

traffic = pd.read_csv(
    "data/interim/traffic_cleaned.csv"
)

weather = pd.read_csv(
    "data/interim/weather_cleaned.csv"
)

calendar = pd.read_csv(
    "data/interim/calendar_cleaned.csv"
)


# --------------------------------------------------
# 2. Convert dates to datetime
# --------------------------------------------------

traffic["date"] = pd.to_datetime(
    traffic["date"],
    errors="coerce"
)

weather["date"] = pd.to_datetime(
    weather["date"],
    dayfirst=True,
    errors="coerce"
)

calendar["date"] = pd.to_datetime(
    calendar["date"],
    errors="coerce"
)


# --------------------------------------------------
# 3. Check invalid dates
# --------------------------------------------------

print("========== INVALID DATE CHECK ==========")

print(
    "Traffic invalid dates:",
    traffic["date"].isna().sum()
)

print(
    "Weather invalid dates:",
    weather["date"].isna().sum()
)

print(
    "Calendar invalid dates:",
    calendar["date"].isna().sum()
)


# --------------------------------------------------
# 4. Display date ranges
# --------------------------------------------------

print("\n========== DATE RANGES ==========")

print(
    "Traffic:",
    traffic["date"].min(),
    "to",
    traffic["date"].max()
)

print(
    "Weather:",
    weather["date"].min(),
    "to",
    weather["date"].max()
)

print(
    "Calendar:",
    calendar["date"].min(),
    "to",
    calendar["date"].max()
)


# --------------------------------------------------
# 5. Save standardized datasets
# --------------------------------------------------

traffic.to_csv(
    "data/interim/traffic_dates.csv",
    index=False
)

weather.to_csv(
    "data/interim/weather_dates.csv",
    index=False
)

calendar.to_csv(
    "data/interim/calendar_dates.csv",
    index=False
)


print("\nDate standardization completed successfully.")