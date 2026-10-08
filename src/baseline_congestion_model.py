import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

print("==================================================")
print("      FLOWCAST BASELINE CONGESTION MODEL")
print("==================================================")

# Load training and validation data
train = pd.read_csv("data/processed/train.csv")
validation = pd.read_csv("data/processed/validation.csv")

print("\nTraining rows:", len(train))
print("Validation rows:", len(validation))

# Features used for the model
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

# Create X and y
X_train = train[features]
y_train = train[target]

X_validation = validation[features]
y_validation = validation[target]

print("\nNumber of features:", len(features))
print("Target:", target)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest model...")

# Train model
model.fit(X_train, y_train)

print("Model training completed.")

# Make predictions
y_pred = model.predict(X_validation)

# Calculate accuracy
accuracy = accuracy_score(y_validation, y_pred)

print("\n========== MODEL PERFORMANCE ==========")
print("Validation Accuracy:", round(accuracy, 4))

# Classification report
print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_validation, y_pred))

# Confusion matrix
print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_validation, y_pred))

# Feature importance
importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance)

# Save model
model_path = "models/baseline_congestion_model.joblib"

joblib.dump(model, model_path)

print("\nModel saved successfully to:")
print(model_path)

print("\n==================================================")
print("      BASELINE MODEL COMPLETED")
print("==================================================")