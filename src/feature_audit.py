import pandas as pd

# ==================================================
# LOAD FEATURE-ENGINEERED DATA
# ==================================================

df = pd.read_csv("data/interim/flowcast_features.csv")

print("==================================================")
print("          FLOWCAST FEATURE AUDIT")
print("==================================================")

# ==================================================
# BASIC INFORMATION
# ==================================================

print("\n========== DATASET ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# ==================================================
# TARGET COLUMN
# ==================================================

target = "congestion_level"

print("\n========== TARGET ==========")
print("Target column:", target)
print("Unique values:")
print(df[target].value_counts())

# ==================================================
# CHECK CONSTANT FEATURES
# ==================================================

print("\n========== CONSTANT FEATURES ==========")

constant_columns = []

for column in df.columns:
    if df[column].nunique() <= 1:
        constant_columns.append(column)

if constant_columns:
    for column in constant_columns:
        print(column)
else:
    print("No constant features found.")

# ==================================================
# CHECK LOW-VARIANCE / LOW-INFORMATION FEATURES
# ==================================================

print("\n========== UNIQUE VALUE COUNTS ==========")

for column in df.columns:
    unique_count = df[column].nunique()

    if unique_count <= 10:
        print(f"{column}: {unique_count} unique values")

# ==================================================
# FEATURES THAT ARE DIRECTLY RELATED TO TARGET
# ==================================================

print("\n========== POTENTIAL TARGET-RELATED FEATURES ==========")

target_related = [
    "traffic_volume",
    "avg_speed",
    "occupancy",
    "road_capacity",
    "volume_capacity_ratio"
]

for column in target_related:
    if column in df.columns:
        print(column)

# ==================================================
# ENGINEERED FEATURES
# ==================================================

print("\n========== ENGINEERED FEATURES ==========")

engineered_features = [
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "day",
    "is_rush_hour",
    "volume_capacity_ratio",
    "speed_capacity_ratio",
    "vehicle_capacity_ratio",
    "accident_flag",
    "is_rain",
    "is_fog",
    "is_poor_visibility",
    "is_extreme_temperature",
    "vehicle_2w_ratio",
    "vehicle_car_ratio",
    "vehicle_lcv_ratio",
    "vehicle_hcv_ratio"
]

for column in engineered_features:
    if column in df.columns:
        print(column)

# ==================================================
# CHECK EXTREME TEMPERATURE FEATURE
# ==================================================

if "is_extreme_temperature" in df.columns:

    print("\n========== EXTREME TEMPERATURE CHECK ==========")

    print(
        df["is_extreme_temperature"]
        .value_counts()
        .sort_index()
    )

# ==================================================
# CHECK POOR VISIBILITY FEATURE
# ==================================================

if "is_poor_visibility" in df.columns:

    print("\n========== POOR VISIBILITY CHECK ==========")

    print(
        df["is_poor_visibility"]
        .value_counts()
        .sort_index()
    )

# ==================================================
# CHECK RUSH HOUR FEATURE
# ==================================================

if "is_rush_hour" in df.columns:

    print("\n========== RUSH HOUR CHECK ==========")

    print(
        df["is_rush_hour"]
        .value_counts()
        .sort_index()
    )

# ==================================================
# FINAL
# ==================================================

print("\n==================================================")
print("          FEATURE AUDIT COMPLETED")
print("==================================================")