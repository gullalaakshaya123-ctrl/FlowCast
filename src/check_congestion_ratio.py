import pandas as pd


# --------------------------------------------------
# 1. Load cleaned traffic data
# --------------------------------------------------

traffic = pd.read_csv(
    "data/interim/traffic_numeric_cleaned.csv"
)


# --------------------------------------------------
# 2. Calculate traffic-to-capacity ratio
# --------------------------------------------------

traffic["traffic_capacity_ratio"] = (
    traffic["traffic_volume"] / traffic["road_capacity"]
)


# --------------------------------------------------
# 3. Compare ratio for each congestion level
# --------------------------------------------------

print("========== TRAFFIC/CAPACITY RATIO ==========")

result = (
    traffic.groupby("congestion_level")["traffic_capacity_ratio"]
    .agg(["count", "mean", "min", "max"])
    .round(3)
)

print(result)