import pandas as pd


# --------------------------------------------------
# 1. Load the raw datasets
# --------------------------------------------------

traffic = pd.read_csv("data/raw/traffic_sensor_log.csv")
weather = pd.read_csv("data/raw/weather_observations.csv")
calendar = pd.read_csv("data/raw/calendar_events.csv")


# --------------------------------------------------
# 2. Basic information
# --------------------------------------------------

print("========== DATASET SHAPES ==========")

print("Traffic:", traffic.shape)
print("Weather:", weather.shape)
print("Calendar:", calendar.shape)


# --------------------------------------------------
# 3. Missing values
# --------------------------------------------------

print("\n========== MISSING VALUES ==========")

print("\nTraffic:")
print(traffic.isnull().sum())

print("\nWeather:")
print(weather.isnull().sum())

print("\nCalendar:")
print(calendar.isnull().sum())


# --------------------------------------------------
# 4. Duplicate rows
# --------------------------------------------------

print("\n========== DUPLICATE ROWS ==========")

print("Traffic duplicates:", traffic.duplicated().sum())
print("Weather duplicates:", weather.duplicated().sum())
print("Calendar duplicates:", calendar.duplicated().sum())


# --------------------------------------------------
# 5. Data types
# --------------------------------------------------

print("\n========== DATA TYPES ==========")

print("\nTraffic:")
print(traffic.dtypes)

print("\nWeather:")
print(weather.dtypes)

print("\nCalendar:")
print(calendar.dtypes)