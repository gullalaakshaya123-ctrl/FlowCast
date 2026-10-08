import pandas as pd
import numpy as np

print("==================================================")
print(" FLOWCAST LSTM TIME-BASED SPLIT")
print("==================================================")

# Load prepared sequences
X = np.load(
    "data/processed/lstm_X.npy"
)

y = np.load(
    "data/processed/lstm_y.npy"
)

metadata = pd.read_csv(
    "data/processed/lstm_metadata.csv"
)

# Convert target datetime
metadata["target_datetime"] = pd.to_datetime(
    metadata["target_datetime"],
    errors="coerce"
)

# Check data
if metadata["target_datetime"].isna().any():
    raise ValueError("Invalid target datetime values found.")

if len(X) != len(y) or len(X) != len(metadata):
    raise ValueError(
        "X, y and metadata lengths do not match."
    )

print("\nTotal sequences:", len(X))

# --------------------------------------------------
# Time-based split
# --------------------------------------------------

train_mask = (
    metadata["target_datetime"] < "2025-04-01"
)

validation_mask = (
    metadata["target_datetime"] >= "2025-04-01"
)

X_train = X[train_mask.values]
y_train = y[train_mask.values]

X_validation = X[validation_mask.values]
y_validation = y[validation_mask.values]

train_metadata = metadata[
    train_mask
].reset_index(drop=True)

validation_metadata = metadata[
    validation_mask
].reset_index(drop=True)

# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n========== TRAINING DATA ==========")

print("Sequences:", len(X_train))
print(
    "Start:",
    train_metadata["target_datetime"].min()
)
print(
    "End:",
    train_metadata["target_datetime"].max()
)

print("\n========== VALIDATION DATA ==========")

print("Sequences:", len(X_validation))
print(
    "Start:",
    validation_metadata["target_datetime"].min()
)
print(
    "End:",
    validation_metadata["target_datetime"].max()
)

# --------------------------------------------------
# Verify time order
# --------------------------------------------------

latest_train = (
    train_metadata["target_datetime"].max()
)

earliest_validation = (
    validation_metadata["target_datetime"].min()
)

print("\n========== TIME ORDER CHECK ==========")

print(
    "Latest training target:",
    latest_train
)

print(
    "Earliest validation target:",
    earliest_validation
)

if latest_train < earliest_validation:
    print("Time order check: PASSED")
else:
    raise ValueError(
        "Time order check FAILED."
    )

# --------------------------------------------------
# Save split data
# --------------------------------------------------

np.save(
    "data/processed/lstm_X_train.npy",
    X_train
)

np.save(
    "data/processed/lstm_y_train.npy",
    y_train
)

np.save(
    "data/processed/lstm_X_validation.npy",
    X_validation
)

np.save(
    "data/processed/lstm_y_validation.npy",
    y_validation
)

train_metadata.to_csv(
    "data/processed/lstm_train_metadata.csv",
    index=False
)

validation_metadata.to_csv(
    "data/processed/lstm_validation_metadata.csv",
    index=False
)

print("\n========== FILES SAVED ==========")

print("data/processed/lstm_X_train.npy")
print("data/processed/lstm_y_train.npy")
print("data/processed/lstm_X_validation.npy")
print("data/processed/lstm_y_validation.npy")
print("data/processed/lstm_train_metadata.csv")
print("data/processed/lstm_validation_metadata.csv")

print("\n==================================================")
print(" LSTM TIME-BASED SPLIT COMPLETED")
print("==================================================")