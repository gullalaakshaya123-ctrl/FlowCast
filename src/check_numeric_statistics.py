import pandas as pd


# Load the traffic dataset
traffic = pd.read_csv(
    "data/interim/traffic_invalid_fixed.csv"
)


# Check numerical columns
columns = [
    "traffic_volume",
    "avg_speed",
    "occupancy"
]


print("========== NUMERICAL STATISTICS ==========")

print(traffic[columns].describe())