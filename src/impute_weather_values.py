import pandas as pd


# --------------------------------------------------
# 1. Load cleaned weather dataset
# --------------------------------------------------

weather = pd.read_csv(
    "data/interim/weather_conditions_cleaned.csv"
)


# --------------------------------------------------
# 2. Display missing values before imputation
# --------------------------------------------------

print("========== BEFORE IMPUTATION ==========")

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
# 3. Calculate medians
# --------------------------------------------------

temperature_median = weather["temperature"].median()

visibility_median = weather["visibility"].median()


print("\nTemperature median:", temperature_median)
print("Visibility median:", visibility_median)


# --------------------------------------------------
# 4. Fill missing values
# --------------------------------------------------

weather["temperature"] = weather[
    "temperature"
].fillna(temperature_median)


weather["visibility"] = weather[
    "visibility"
].fillna(visibility_median)


# --------------------------------------------------
# 5. Check missing values after imputation
# --------------------------------------------------

print("\n========== AFTER IMPUTATION ==========")

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
# 6. Save cleaned weather dataset
# --------------------------------------------------

weather.to_csv(
    "data/interim/weather_cleaned.csv",
    index=False
)


print(
    "\nWeather missing values handled successfully."
)

print(
    "Final cleaned weather dataset saved successfully."
)