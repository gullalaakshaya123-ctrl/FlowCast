import pandas as pd
import numpy as np


# --------------------------------------------------
# 1. Load the deduplicated traffic dataset
# --------------------------------------------------

traffic = pd.read_csv("data/interim/traffic_deduplicated.csv")


# --------------------------------------------------
# 2. Count invalid values before fixing
# --------------------------------------------------

negative_volume = (traffic["traffic_volume"] < 0).sum()
invalid_speed = (traffic["avg_speed"] > 200).sum()
invalid_occupancy = (traffic["occupancy"] > 100).sum()

print("Invalid values before fixing:")
print("Negative traffic volume:", negative_volume)
print("Speed above 200 km/h:", invalid_speed)
print("Occupancy above 100%:", invalid_occupancy)


# --------------------------------------------------
# 3. Replace invalid values with NaN
# --------------------------------------------------

traffic.loc[traffic["traffic_volume"] < 0, "traffic_volume"] = np.nan

traffic.loc[traffic["avg_speed"] > 200, "avg_speed"] = np.nan

traffic.loc[traffic["occupancy"] > 100, "occupancy"] = np.nan


# --------------------------------------------------
# 4. Check the result
# --------------------------------------------------

print("\nInvalid values after fixing:")

print(
    "Negative traffic volume:",
    (traffic["traffic_volume"] < 0).sum()
)

print(
    "Speed above 200 km/h:",
    (traffic["avg_speed"] > 200).sum()
)

print(
    "Occupancy above 100%:",
    (traffic["occupancy"] > 100).sum()
)


# --------------------------------------------------
# 5. Save the cleaned dataset
# --------------------------------------------------

traffic.to_csv(
    "data/interim/traffic_invalid_fixed.csv",
    index=False
)

print("\nInvalid values replaced with NaN.")
print("Cleaned dataset saved successfully.")