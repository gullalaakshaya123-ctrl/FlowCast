import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("==================================================")
print(" ORIGINAL-LABEL XGBOOST EVALUATION")
print("==================================================")

# Load original-label train and validation data
train = pd.read_csv("data/processed/original_train.csv")
validation = pd.read_csv("data/processed/original_validation.csv")

# Features used by the XGBoost model
features = [
    "traffic_volume",
    "avg_speed",
    "occupancy",
    "road_capacity",
    "vehicle_count",
    "accident_count",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_rush_hour",
    "is_rain",
    "is_fog",
    "is_poor_visibility",
    "vehicle_2w_ratio",
    "vehicle_car_ratio",
    "vehicle_lcv_ratio",
    "vehicle_hcv_ratio"
]

target = "congestion_level"

# Prepare training and validation data
X_train = train[features]
y_train = train[target]

X_validation = validation[features]
y_validation = validation[target]

print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_validation))

# Load the trained XGBoost model
model = joblib.load("models/xgboost_congestion_model.joblib")

# Predict validation data
y_pred = model.predict(X_validation)

# Convert numeric predictions back to class names
label_mapping = {
    0: "Free-flow",
    1: "Moderate",
    2: "Heavy",
    3: "Severe"
}

y_pred = pd.Series(y_pred).map(label_mapping)

# Accuracy
accuracy = accuracy_score(y_validation, y_pred)

print("\n========== VALIDATION ACCURACY ==========")
print(f"Accuracy: {accuracy:.4f}")

# Classification report
print("\n========== CLASSIFICATION REPORT ==========")
print(
    classification_report(
        y_validation,
        y_pred,
        digits=4
    )
)

print("\n==================================================")
print(" ORIGINAL-LABEL XGBOOST EVALUATION COMPLETED")
print("==================================================")