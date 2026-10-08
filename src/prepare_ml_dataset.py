import pandas as pd

# ==================================================
# LOAD FEATURE-ENGINEERED DATA
# ==================================================

df = pd.read_csv("data/interim/flowcast_features.csv")

print("==================================================")
print("       FLOWCAST ML DATASET PREPARATION")
print("==================================================")

print("\nOriginal shape:")
print(df.shape)

# ==================================================
# REMOVE USELESS / CONSTANT FEATURES
# ==================================================

columns_to_remove = [
    "is_extreme_temperature"
]

df_ml = df.drop(columns=columns_to_remove)

print("\nRemoved columns:")
for column in columns_to_remove:
    print("-", column)

# ==================================================
# CHECK RESULT
# ==================================================

print("\nNew shape:")
print(df_ml.shape)

print("\nRemaining missing values:")
print(df_ml.isnull().sum().sum())

print("\nRemaining duplicate rows:")
print(df_ml.duplicated().sum())

# ==================================================
# SAVE ML DATASET
# ==================================================

output_path = "data/processed/flowcast_ml_ready.csv"

df_ml.to_csv(output_path, index=False)

print("\nML-ready dataset saved to:")
print(output_path)

print("\n==================================================")
print("       ML DATASET PREPARATION COMPLETED")
print("==================================================")