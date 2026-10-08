import pandas as pd


# --------------------------------------------------
# 1. Load the deduplicated traffic dataset
# --------------------------------------------------

traffic = pd.read_csv("data/interim/traffic_deduplicated.csv")


# --------------------------------------------------
# 2. Check negative traffic volume
# --------------------------------------------------

negative_volume = (traffic["traffic_volume"] < 0).sum()

print("Negative traffic volume:", negative_volume)


# --------------------------------------------------
# 3. Check impossible speed values
# --------------------------------------------------

invalid_speed = (traffic["avg_speed"] > 200).sum()

print("Speed values above 200 km/h:", invalid_speed)


# --------------------------------------------------
# 4. Check invalid occupancy values
# --------------------------------------------------

invalid_occupancy = (traffic["occupancy"] > 100).sum()

print("Occupancy values above 100%:", invalid_occupancy)


# --------------------------------------------------
# 5. Display some invalid records
# --------------------------------------------------

print("\n========== SAMPLE INVALID VALUES ==========")

print("\nNegative traffic volume:")
print(traffic[traffic["traffic_volume"] < 0][
    ["road_id", "date", "time", "traffic_volume"]
].head())


print("\nSpeed above 200 km/h:")
print(traffic[traffic["avg_speed"] > 200][
    ["road_id", "date", "time", "avg_speed"]
].head())


print("\nOccupancy above 100%:")
print(traffic[traffic["occupancy"] > 100][
    ["road_id", "date", "time", "occupancy"]
].head())