import pandas as pd


# --------------------------------------------------
# 1. Load weather dataset
# --------------------------------------------------

weather = pd.read_csv(
    "data/raw/weather_observations.csv"
)


# --------------------------------------------------
# 2. Basic information
# --------------------------------------------------

print("========== WEATHER DATASET ==========")

print("Rows:", len(weather))
print("Columns:", len(weather.columns))

print("\nColumns:")
print(list(weather.columns))


# --------------------------------------------------
# 3. Missing values
# --------------------------------------------------

print("\n========== MISSING VALUES ==========")

print(
    weather.isnull().sum()
)


# --------------------------------------------------
# 4. Duplicate records
# --------------------------------------------------

print("\n========== DUPLICATES ==========")

print(
    "Duplicate rows:",
    weather.duplicated().sum()
)


# --------------------------------------------------
# 5. Data types
# --------------------------------------------------

print("\n========== DATA TYPES ==========")

print(
    weather.dtypes
)


# --------------------------------------------------
# 6. Weather conditions
# --------------------------------------------------

print("\n========== WEATHER CONDITIONS ==========")

print(
    weather["weather_condition"].value_counts(
        dropna=False
    )
)


# --------------------------------------------------
# 7. Date range
# --------------------------------------------------

print("\n========== DATE RANGE ==========")

print(
    "First date:",
    weather["date"].min()
)

print(
    "Last date:",
    weather["date"].max()
)