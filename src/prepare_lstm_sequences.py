import pandas as pd
import numpy as np

print("==================================================")
print(" FLOWCAST LSTM SEQUENCE PREPARATION")
print("==================================================")

# --------------------------------------------------
# 1. Load feature dataset
# --------------------------------------------------

df = pd.read_csv(
    "data/interim/flowcast_features.csv"
)

# Convert datetime
df["datetime"] = pd.to_datetime(
    df["datetime"],
    errors="coerce"
)

# Check datetime
if df["datetime"].isna().any():
    raise ValueError("Invalid datetime values found.")

# Sort by road and time
df = df.sort_values(
    ["road_id", "datetime"]
).reset_index(drop=True)

# --------------------------------------------------
# 2. LSTM configuration
# --------------------------------------------------

SEQUENCE_LENGTH = 12

# 12 × 30 minutes = 6 hours
print("\nSequence length:", SEQUENCE_LENGTH)
print("History used: 6 hours")

# Features used by LSTM
features = [
    "traffic_volume",
    "avg_speed",
    "occupancy",
    "vehicle_count",
    "road_capacity"
]

target = "traffic_volume"

# --------------------------------------------------
# 3. Check required columns
# --------------------------------------------------

required_columns = features + [
    target,
    "road_id",
    "datetime"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# --------------------------------------------------
# 4. Create sequences
# --------------------------------------------------

X_sequences = []
y_values = []
sequence_times = []
sequence_roads = []

skipped_sequences = 0

# Process every road separately
for road_id, road_data in df.groupby("road_id"):

    road_data = (
        road_data
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # Calculate time difference
    road_data["time_difference"] = (
        road_data["datetime"].diff()
    )

    for i in range(
        SEQUENCE_LENGTH,
        len(road_data)
    ):

        # Previous 12 observations
        sequence = road_data.iloc[
            i - SEQUENCE_LENGTH:i
        ]

        # Current observation = target
        current = road_data.iloc[i]

        # --------------------------------------------------
        # Check that all 12 historical observations
        # are exactly 30 minutes apart
        # --------------------------------------------------

        time_differences = (
            sequence["datetime"]
            .diff()
            .dropna()
        )

        if not (
            time_differences
            == pd.Timedelta(minutes=30)
        ).all():
            skipped_sequences += 1
            continue

        # Also make sure target follows the sequence
        # by exactly 30 minutes
        last_sequence_time = sequence[
            "datetime"
        ].iloc[-1]

        target_time = current["datetime"]

        if (
            target_time - last_sequence_time
            != pd.Timedelta(minutes=30)
        ):
            skipped_sequences += 1
            continue

        # --------------------------------------------------
        # Store sequence
        # --------------------------------------------------

        X_sequences.append(
            sequence[features].values
        )

        y_values.append(
            current[target]
        )

        sequence_times.append(
            target_time
        )

        sequence_roads.append(
            road_id
        )

# --------------------------------------------------
# 5. Convert to NumPy arrays
# --------------------------------------------------

X = np.array(
    X_sequences,
    dtype=np.float32
)

y = np.array(
    y_values,
    dtype=np.float32
)

# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

print("\n========== LSTM DATA ==========")

print("Total sequences:", len(X))
print("Sequence shape:", X.shape)
print("Target shape:", y.shape)

print(
    "Skipped sequences due to time gaps:",
    skipped_sequences
)

print(
    "Sequence length:",
    SEQUENCE_LENGTH
)

print(
    "Number of features:",
    len(features)
)

print("\nFeatures:")

for feature in features:
    print("-", feature)

print("\nTarget:")
print("-", target)

# --------------------------------------------------
# 7. Save sequences
# --------------------------------------------------

np.save(
    "data/processed/lstm_X.npy",
    X
)

np.save(
    "data/processed/lstm_y.npy",
    y
)

# --------------------------------------------------
# 8. Save sequence metadata
# --------------------------------------------------

metadata = pd.DataFrame({
    "road_id": sequence_roads,
    "target_datetime": sequence_times
})

metadata.to_csv(
    "data/processed/lstm_metadata.csv",
    index=False
)

print("\n========== FILES SAVED ==========")

print(
    "data/processed/lstm_X.npy"
)

print(
    "data/processed/lstm_y.npy"
)

print(
    "data/processed/lstm_metadata.csv"
)

print("\n==================================================")
print(" LSTM SEQUENCE PREPARATION COMPLETED")
print("==================================================")