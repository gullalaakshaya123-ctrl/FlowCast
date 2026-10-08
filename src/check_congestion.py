import pandas as pd


# --------------------------------------------------
# 1. Load the cleaned traffic dataset
# --------------------------------------------------

traffic = pd.read_csv(
    "data/interim/traffic_numeric_cleaned.csv"
)


# --------------------------------------------------
# 2. Check congestion level counts
# --------------------------------------------------

print("========== CONGESTION LEVEL COUNTS ==========")

print(
    traffic["congestion_level"].value_counts(dropna=False)
)


# --------------------------------------------------
# 3. Check congestion level percentages
# --------------------------------------------------

print("\n========== CONGESTION LEVEL PERCENTAGES ==========")

print(
    (traffic["congestion_level"]
     .value_counts(normalize=True, dropna=False) * 100)
    .round(2)
)