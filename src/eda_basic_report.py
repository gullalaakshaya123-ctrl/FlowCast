import pandas as pd


# ==================================================
# LOAD FEATURE-ENGINEERED DATA
# ==================================================

df = pd.read_csv(
    "data/interim/flowcast_features.csv"
)


# ==================================================
# BASIC DATASET INFORMATION
# ==================================================

print("==================================================")
print("          FLOWCAST BASIC EDA REPORT")
print("==================================================")


print("\n========== DATASET SHAPE ==========")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ==================================================
# COLUMN INFORMATION
# ==================================================

print("\n========== COLUMN NAMES ==========")

for column in df.columns:
    print(column)


# ==================================================
# DATA TYPES
# ==================================================

print("\n========== DATA TYPES ==========")

print(df.dtypes)


# ==================================================
# MISSING VALUES
# ==================================================

print("\n========== MISSING VALUES ==========")

missing_values = df.isnull().sum()

print(missing_values)


# ==================================================
# DUPLICATE ROWS
# ==================================================

print("\n========== DUPLICATE ROWS ==========")

print(
    "Duplicate rows:",
    df.duplicated().sum()
)


# ==================================================
# NUMERIC STATISTICS
# ==================================================

print("\n========== NUMERIC STATISTICS ==========")

print(
    df.describe()
)


# ==================================================
# CONGESTION DISTRIBUTION
# ==================================================

print("\n========== CONGESTION DISTRIBUTION ==========")

congestion_counts = (
    df["congestion_level"]
    .value_counts()
)

print(congestion_counts)


print("\n========== CONGESTION PERCENTAGE ==========")

congestion_percentage = (
    df["congestion_level"]
    .value_counts(normalize=True)
    * 100
)

print(
    congestion_percentage.round(2)
)


# ==================================================
# WEATHER DISTRIBUTION
# ==================================================

print("\n========== WEATHER DISTRIBUTION ==========")

print(
    df["weather_condition"].value_counts()
)


# ==================================================
# ROAD DISTRIBUTION
# ==================================================

print("\n========== ROAD DISTRIBUTION ==========")

print(
    "Number of unique roads:",
    df["road_id"].nunique()
)


# ==================================================
# DATE RANGE
# ==================================================

print("\n========== DATE RANGE ==========")

print(
    "Start date:",
    df["datetime"].min()
)

print(
    "End date:",
    df["datetime"].max()
)


# ==================================================
# FEATURED DATASET CHECK
# ==================================================

print("\n========== FINAL QUALITY CHECK ==========")

print(
    "Total records:",
    len(df)
)

print(
    "Total columns:",
    len(df.columns)
)

print(
    "Total missing values:",
    df.isnull().sum().sum()
)

print(
    "Total duplicate rows:",
    df.duplicated().sum()
)


print("\n==================================================")
print("          BASIC EDA REPORT COMPLETED")
print("==================================================")