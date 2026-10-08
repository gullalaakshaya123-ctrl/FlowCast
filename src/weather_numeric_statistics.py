import pandas as pd


# --------------------------------------------------
# 1. Load cleaned weather dataset
# --------------------------------------------------

weather = pd.read_csv(
    "data/interim/weather_conditions_cleaned.csv"
)


# --------------------------------------------------
# 2. Display numeric statistics
# --------------------------------------------------

print("========== WEATHER NUMERIC STATISTICS ==========")

print(
    weather[
        [
            "temperature",
            "rainfall",
            "visibility"
        ]
    ].describe()
)


# --------------------------------------------------
# 3. Check missing values
# --------------------------------------------------

print("\n========== MISSING VALUES ==========")

print(
    weather[
        [
            "temperature",
            "rainfall",
            "visibility"
        ]
    ].isnull().sum()
)


# --------------------------------------------------
# 4. Display medians
# --------------------------------------------------

print("\n========== MEDIAN VALUES ==========")

print(
    "Temperature median:",
    weather["temperature"].median()
)

print(
    "Rainfall median:",
    weather["rainfall"].median()
)

print(
    "Visibility median:",
    weather["visibility"].median()
)