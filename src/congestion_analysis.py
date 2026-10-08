import pandas as pd


# --------------------------------------------------
# 1. Load cleaned traffic data
# --------------------------------------------------

traffic = pd.read_csv(
    "data/interim/traffic_numeric_cleaned.csv"
)


# --------------------------------------------------
# 2. Compare traffic measurements by congestion level
# --------------------------------------------------

columns = [
    "traffic_volume",
    "avg_speed",
    "occupancy",
    "road_capacity"
]

print("========== CONGESTION LEVEL ANALYSIS ==========")

print(
    traffic.groupby("congestion_level")[columns]
    .mean()
    .round(2)
)


# --------------------------------------------------
# 3. Count records for each level
# --------------------------------------------------

print("\n========== RECORD COUNTS ==========")

print(
    traffic["congestion_level"].value_counts()
)