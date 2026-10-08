import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# 1. Load cleaned traffic data
# --------------------------------------------------

traffic = pd.read_csv(
    "data/interim/traffic_numeric_cleaned.csv"
)


# --------------------------------------------------
# 2. Define features
# --------------------------------------------------

features = [
    "traffic_volume",
    "avg_speed",
    "occupancy",
    "road_capacity"
]


# --------------------------------------------------
# 3. Separate known and missing congestion labels
# --------------------------------------------------

known = traffic[
    traffic["congestion_level"].notna()
].copy()

missing = traffic[
    traffic["congestion_level"].isna()
].copy()


print("Known congestion records:", len(known))
print("Missing congestion records:", len(missing))


# --------------------------------------------------
# 4. Prepare training data
# --------------------------------------------------

X_train = known[features]
y_train = known["congestion_level"]


# --------------------------------------------------
# 5. Create and train the model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 6. Predict missing congestion labels
# --------------------------------------------------

print("Predicting missing congestion labels...")

predictions = model.predict(
    missing[features]
)


# --------------------------------------------------
# 7. Fill only missing values
# --------------------------------------------------

traffic.loc[
    traffic["congestion_level"].isna(),
    "congestion_level"
] = predictions


# --------------------------------------------------
# 8. Check remaining missing values
# --------------------------------------------------

remaining_missing = traffic[
    "congestion_level"
].isna().sum()

print(
    "\nRemaining missing congestion labels:",
    remaining_missing
)


# --------------------------------------------------
# 9. Display final distribution
# --------------------------------------------------

print("\n========== FINAL CONGESTION DISTRIBUTION ==========")

print(
    traffic["congestion_level"].value_counts()
)


# --------------------------------------------------
# 10. Save cleaned dataset
# --------------------------------------------------

traffic.to_csv(
    "data/interim/traffic_cleaned.csv",
    index=False
)

print("\nCongestion labels handled successfully.")
print("Final cleaned traffic dataset saved successfully.")