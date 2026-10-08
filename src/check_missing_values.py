import pandas as pd


# --------------------------------------------------
# 1. Load the traffic dataset after invalid-value fixing
# --------------------------------------------------

traffic = pd.read_csv("data/interim/traffic_invalid_fixed.csv")


# --------------------------------------------------
# 2. Check missing values
# --------------------------------------------------

print("========== MISSING VALUES ==========")

print(traffic.isnull().sum())


# --------------------------------------------------
# 3. Show percentage of missing values
# --------------------------------------------------

print("\n========== MISSING VALUE PERCENTAGE ==========")

missing_percentage = (
    traffic.isnull().sum() / len(traffic) * 100
).round(2)

print(missing_percentage)