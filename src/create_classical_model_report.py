import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

print("==================================================")
print(" FLOWCAST CLASSICAL MODEL REPORT")
print("==================================================")

# Load original-label validation data
validation = pd.read_csv(
    "data/processed/original_validation.csv"
)

# Features used by both models
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

X_validation = validation[features]
y_validation = validation[target]

# Class order
class_names = [
    "Free-flow",
    "Moderate",
    "Heavy",
    "Severe"
]

# ==================================================
# RANDOM FOREST
# ==================================================

print("\nLoading Random Forest model...")

rf_model = joblib.load(
    "models/baseline_congestion_model.joblib"
)

rf_predictions = rf_model.predict(X_validation)

rf_accuracy = accuracy_score(
    y_validation,
    rf_predictions
)

rf_report = classification_report(
    y_validation,
    rf_predictions,
    output_dict=True
)

print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# Random Forest confusion matrix
rf_cm = confusion_matrix(
    y_validation,
    rf_predictions,
    labels=class_names
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=rf_cm,
    display_labels=class_names
)

disp.plot()
plt.title("Random Forest - Congestion Level Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "reports/figures/rf_confusion_matrix.png",
    dpi=300
)

plt.close()

print("Random Forest confusion matrix saved.")


# ==================================================
# XGBOOST
# ==================================================

print("\nLoading XGBoost model...")

xgb_model = joblib.load(
    "models/xgboost_congestion_model.joblib"
)

xgb_predictions = xgb_model.predict(X_validation)

# Convert numeric predictions to labels
label_mapping = {
    0: "Free-flow",
    1: "Moderate",
    2: "Heavy",
    3: "Severe"
}

xgb_predictions = pd.Series(
    xgb_predictions
).map(label_mapping)

xgb_accuracy = accuracy_score(
    y_validation,
    xgb_predictions
)

xgb_report = classification_report(
    y_validation,
    xgb_predictions,
    output_dict=True
)

print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

# XGBoost confusion matrix
xgb_cm = confusion_matrix(
    y_validation,
    xgb_predictions,
    labels=class_names
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=xgb_cm,
    display_labels=class_names
)

disp.plot()
plt.title("XGBoost - Congestion Level Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "reports/figures/xgb_confusion_matrix.png",
    dpi=300
)

plt.close()

print("XGBoost confusion matrix saved.")


# ==================================================
# MODEL COMPARISON
# ==================================================

comparison = pd.DataFrame({
    "Model": [
        "Random Forest",
        "XGBoost"
    ],
    "Accuracy": [
        rf_accuracy,
        xgb_accuracy
    ],
    "Macro_F1": [
        rf_report["macro avg"]["f1-score"],
        xgb_report["macro avg"]["f1-score"]
    ],
    "Weighted_F1": [
        rf_report["weighted avg"]["f1-score"],
        xgb_report["weighted avg"]["f1-score"]
    ]
})

print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

comparison.to_csv(
    "reports/classical_model_comparison.csv",
    index=False
)

print("\nModel comparison saved to:")
print("reports/classical_model_comparison.csv")

print("\n==================================================")
print(" CLASSICAL MODEL REPORT COMPLETED")
print("==================================================")