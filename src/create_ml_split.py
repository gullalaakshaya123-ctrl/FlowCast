import pandas as pd

# ==================================================
# LOAD ML-READY DATA
# ==================================================

df = pd.read_csv("data/processed/flowcast_ml_ready.csv")

# Convert datetime to actual datetime
df["datetime"] = pd.to_datetime(df["datetime"])

print("==================================================")
print("          FLOWCAST TIME-BASED ML SPLIT")
print("==================================================")

print("\nTotal records:", len(df))

# ==================================================
# SORT BY TIME
# ==================================================

df = df.sort_values("datetime").reset_index(drop=True)

# ==================================================
# TIME-BASED SPLIT
# ==================================================
# Training:
# January 2025 to March 2025
#
# Validation:
# April 2025 to May 2025

train = df[df["datetime"] < "2025-04-01"].copy()

validation = df[df["datetime"] >= "2025-04-01"].copy()

# ==================================================
# DISPLAY SPLIT INFORMATION
# ==================================================

print("\n========== TRAINING DATA ==========")

print("Rows:", len(train))
print("Start:", train["datetime"].min())
print("End:", train["datetime"].max())

print("\n========== VALIDATION DATA ==========")

print("Rows:", len(validation))
print("Start:", validation["datetime"].min())
print("End:", validation["datetime"].max())

# ==================================================
# CHECK FOR OVERLAP
# ==================================================

latest_train = train["datetime"].max()
earliest_validation = validation["datetime"].min()

print("\n========== TIME ORDER CHECK ==========")

print("Latest training timestamp:", latest_train)
print("Earliest validation timestamp:", earliest_validation)

if latest_train < earliest_validation:
    print("Time order check: PASSED")
else:
    print("Time order check: FAILED")

# ==================================================
# SAVE SPLITS
# ==================================================

train.to_csv(
    "data/processed/train.csv",
    index=False
)

validation.to_csv(
    "data/processed/validation.csv",
    index=False
)

print("\nTraining data saved to:")
print("data/processed/train.csv")

print("\nValidation data saved to:")
print("data/processed/validation.csv")

print("\n==================================================")
print("          ML SPLIT COMPLETED")
print("==================================================")