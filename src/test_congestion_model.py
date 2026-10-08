import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. Load cleaned traffic data
# --------------------------------------------------

traffic = pd.read_csv(
    "data/interim/traffic_numeric_cleaned.csv"
)


# --------------------------------------------------
# 2. Keep only rows with known congestion labels
# --------------------------------------------------

known = traffic.dropna(
    subset=["congestion_level"]
).copy()


# --------------------------------------------------
# 3. Select input features
# --------------------------------------------------

features = [
    "traffic_volume",
    "avg_speed",
    "occupancy",
    "road_capacity"
]

X = known[features]
y = known["congestion_level"]


# --------------------------------------------------
# 4. Split data into training and validation sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 5. Create the classification model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 6. Train the model
# --------------------------------------------------

print("Training model...")

model.fit(X_train, y_train)


# --------------------------------------------------
# 7. Make predictions on validation data
# --------------------------------------------------

predictions = model.predict(X_test)


# --------------------------------------------------
# 8. Evaluate the model
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n========== MODEL RESULTS ==========")

print("Validation Accuracy:", round(accuracy, 4))


print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions
    )
)