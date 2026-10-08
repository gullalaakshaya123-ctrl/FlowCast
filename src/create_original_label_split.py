import pandas as pd

print("==================================================")
print(" FLOWCAST ORIGINAL-LABEL FEATURED TIME-BASED SPLIT")
print("==================================================")

# Load fully feature-engineered dataset
df = pd.read_csv(
    "data/interim/flowcast_features.csv"
)

# Load label provenance
provenance = pd.read_csv(
    "data/interim/traffic_cleaned_with_provenance.csv"
)

print("\nFeature dataset rows:", len(df))
print("Provenance dataset rows:", len(provenance))

# Make sure row counts match
if len(df) != len(provenance):
    raise ValueError("Feature and provenance row counts do not match.")

# Add provenance information
df["congestion_label_source"] = (
    provenance["congestion_label_source"].values
)

# Keep only originally observed labels
df = df[
    df["congestion_label_source"] == "original"
].copy()

print("\nOriginal-label records:", len(df))

# Convert datetime
df["datetime"] = pd.to_datetime(
    df["datetime"],
    errors="coerce"
)

# Check datetime
invalid_datetime = df["datetime"].isna().sum()

print("Invalid datetime values:", invalid_datetime)

if invalid_datetime > 0:
    raise ValueError("Invalid datetime values found.")

# Sort chronologically
df = df.sort_values("datetime").reset_index(drop=True)

# Time-based split
train = df[
    df["datetime"] < "2025-04-01"
].copy()

validation = df[
    df["datetime"] >= "2025-04-01"
].copy()

print("\n========== TRAINING DATA ==========")
print("Rows:", len(train))
print("Start:", train["datetime"].min())
print("End:", train["datetime"].max())

print("\n========== VALIDATION DATA ==========")
print("Rows:", len(validation))
print("Start:", validation["datetime"].min())
print("End:", validation["datetime"].max())

# Time-order check
latest_train = train["datetime"].max()
earliest_validation = validation["datetime"].min()

print("\n========== TIME ORDER CHECK ==========")
print("Latest training timestamp:", latest_train)
print("Earliest validation timestamp:", earliest_validation)

if latest_train < earliest_validation:
    print("Time order check: PASSED")
else:
    print("Time order check: FAILED")

# Label distributions
print("\n========== TRAINING LABELS ==========")
print(train["congestion_level"].value_counts())

print("\n========== VALIDATION LABELS ==========")
print(validation["congestion_level"].value_counts())

# Save
train.to_csv(
    "data/processed/original_train.csv",
    index=False
)

validation.to_csv(
    "data/processed/original_validation.csv",
    index=False
)

print("\nTraining data saved to:")
print("data/processed/original_train.csv")

print("\nValidation data saved to:")
print("data/processed/original_validation.csv")

print("\n==================================================")
print(" FEATURED ORIGINAL-LABEL SPLIT COMPLETED")
print("==================================================")