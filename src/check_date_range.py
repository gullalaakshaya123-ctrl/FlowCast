import pandas as pd


# Load cleaned traffic data
traffic = pd.read_csv(
    "data/interim/traffic_numeric_cleaned.csv"
)


# Convert date to datetime
traffic["date"] = pd.to_datetime(
    traffic["date"]
)


# Display date range
print("========== DATE RANGE ==========")

print("First date:", traffic["date"].min())
print("Last date:", traffic["date"].max())


# Display number of records by month
print("\n========== RECORDS BY MONTH ==========")

print(
    traffic["date"]
    .dt.to_period("M")
    .value_counts()
    .sort_index()
)