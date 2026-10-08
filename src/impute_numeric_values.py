import pandas as pd


# --------------------------------------------------
# 1. Load the traffic dataset
# --------------------------------------------------

traffic = pd.read_csv(
    "data/interim/traffic_invalid_fixed.csv"
)


# --------------------------------------------------
# 2. Numerical columns to clean
# --------------------------------------------------

numeric_columns = [
    "traffic_volume",
    "avg_speed",
    "occupancy"
]


# --------------------------------------------------
# 3. Display missing values before imputation
# --------------------------------------------------

print("========== BEFORE IMPUTATION ==========")

print(traffic[numeric_columns].isnull().sum())


# --------------------------------------------------
# 4. Fill missing values using median
# --------------------------------------------------

for column in numeric_columns:
    median_value = traffic[column].median()

    traffic[column] = traffic[column].fillna(median_value)

    print(
        f"{column}: missing values filled with median = {median_value}"
    )


# --------------------------------------------------
# 5. Check missing values after imputation
# --------------------------------------------------

print("\n========== AFTER IMPUTATION ==========")

print(traffic[numeric_columns].isnull().sum())


# --------------------------------------------------
# 6. Save the cleaned dataset
# --------------------------------------------------

traffic.to_csv(
    "data/interim/traffic_numeric_cleaned.csv",
    index=False
)

print("\nNumeric missing values handled successfully.")
print("Cleaned dataset saved successfully.")