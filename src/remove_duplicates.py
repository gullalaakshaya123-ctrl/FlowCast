import pandas as pd


# --------------------------------------------------
# 1. Load the traffic dataset
# --------------------------------------------------

traffic = pd.read_csv("data/raw/traffic_sensor_log.csv")


# --------------------------------------------------
# 2. Check duplicate rows before removal
# --------------------------------------------------

duplicates_before = traffic.duplicated().sum()

print("Duplicate rows before cleaning:", duplicates_before)


# --------------------------------------------------
# 3. Remove exact duplicate rows
# --------------------------------------------------

traffic_cleaned = traffic.drop_duplicates().copy()


# --------------------------------------------------
# 4. Check duplicate rows after removal
# --------------------------------------------------

duplicates_after = traffic_cleaned.duplicated().sum()

print("Duplicate rows after cleaning:", duplicates_after)


# --------------------------------------------------
# 5. Compare dataset size
# --------------------------------------------------

print("\nOriginal traffic rows:", len(traffic))
print("Cleaned traffic rows:", len(traffic_cleaned))
print("Rows removed:", len(traffic) - len(traffic_cleaned))


# --------------------------------------------------
# 6. Save cleaned traffic dataset
# --------------------------------------------------

traffic_cleaned.to_csv(
    "data/interim/traffic_deduplicated.csv",
    index=False
)

print("\nCleaned traffic dataset saved successfully.")