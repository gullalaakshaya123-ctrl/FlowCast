import pandas as pd


# --------------------------------------------------
# 1. Load raw weather dataset
# --------------------------------------------------

weather = pd.read_csv(
    "data/raw/weather_observations.csv"
)


# --------------------------------------------------
# 2. Display conditions before cleaning
# --------------------------------------------------

print("========== BEFORE CLEANING ==========")

print(
    weather["weather_condition"].value_counts()
)


# --------------------------------------------------
# 3. Standardize weather conditions
# --------------------------------------------------

weather["weather_condition"] = (
    weather["weather_condition"]
    .str.strip()
    .str.lower()
)


weather["weather_condition"] = (
    weather["weather_condition"]
    .replace({
        "clear": "Clear",
        "cloudy": "Cloudy",
        "rain": "Rain",
        "rainy": "Rain",
        "fog": "Fog",
        "foggy": "Fog",
        "overcast": "Overcast"
    })
)


# --------------------------------------------------
# 4. Display conditions after cleaning
# --------------------------------------------------

print("\n========== AFTER CLEANING ==========")

print(
    weather["weather_condition"].value_counts()
)


# --------------------------------------------------
# 5. Save cleaned weather dataset
# --------------------------------------------------

weather.to_csv(
    "data/interim/weather_conditions_cleaned.csv",
    index=False
)


print(
    "\nWeather conditions cleaned successfully."
)

print(
    "Cleaned weather dataset saved successfully."
)