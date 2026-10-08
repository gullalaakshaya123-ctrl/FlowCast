import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. Load cleaned traffic data
# --------------------------------------------------

traffic = pd.read_csv(
    "data/interim/traffic_numeric_cleaned.csv"
)


# --------------------------------------------------
# 2. Convert date to datetime
# --------------------------------------------------

traffic["date"] = pd.to_datetime(
    traffic["date"]
)


# --------------------------------------------------
# 3. Keep only records with known congestion labels
# --------------------------------------------------

known = traffic.dropna(
    subset=["congestion_level"]
).copy()


# --------------------------------------------------
# 4. Define features
# --------------------------------------------------

features = [
    "traffic_volume",
    "avg_speed",
    "occupancy",
    "road_capacity"
]


# --------------------------------------------------
# 5. Create time-based train and validation sets
# --------------------------------------------------

train = known[
    known["date"] < "2025-04-01"
]

validation = known[
    known["date"] >= "2025-04-01"
]


X_train = train[features]
y_train = train["congestion_level"]

X_validation = validation[features]
y_validation = validation["congestion_level"]


print("Training records:", len(train))
print("Validation records:", len(validation))


# --------------------------------------------------
# 6. Create Random Forest classifier
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 7. Train model
# --------------------------------------------------

print("\nTraining time-based model...")

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 8. Predict validation data
# --------------------------------------------------

predictions = model.predict(
    X_validation
)


# --------------------------------------------------
# 9. Evaluate model
# --------------------------------------------------

accuracy = accuracy_score(
    y_validation,
    predictions
)


print("\n========== TIME-BASED MODEL RESULTS ==========")

print(
    "Validation Accuracy:",
    round(accuracy, 4)
)


print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_validation,
        predictions
    )
)