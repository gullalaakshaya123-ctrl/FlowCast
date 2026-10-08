import pandas as pd

print("==================================================")
print(" FLOWCAST MODEL COMPARISON REPORT")
print("==================================================")

# Classical ML results
xgboost_accuracy = 0.9927
xgboost_macro_f1 = 0.9860
xgboost_weighted_f1 = 0.9927

random_forest_accuracy = 0.9836
random_forest_macro_f1 = 0.9709
random_forest_weighted_f1 = 0.9837

# LSTM results
lstm_mae = 54.9857
lstm_rmse = 85.4655
lstm_r2 = 0.9062

# Create comparison table
comparison = pd.DataFrame([
    {
        "Model": "Random Forest",
        "Task": "Congestion Classification",
        "Accuracy": random_forest_accuracy,
        "Macro F1": random_forest_macro_f1,
        "Weighted F1": random_forest_weighted_f1,
        "MAE": None,
        "RMSE": None,
        "R2": None
    },
    {
        "Model": "XGBoost",
        "Task": "Congestion Classification",
        "Accuracy": xgboost_accuracy,
        "Macro F1": xgboost_macro_f1,
        "Weighted F1": xgboost_weighted_f1,
        "MAE": None,
        "RMSE": None,
        "R2": None
    },
    {
        "Model": "LSTM",
        "Task": "Traffic Volume Forecasting",
        "Accuracy": None,
        "Macro F1": None,
        "Weighted F1": None,
        "MAE": lstm_mae,
        "RMSE": lstm_rmse,
        "R2": lstm_r2
    }
])

# Save report
comparison.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\n========== MODEL COMPARISON ==========")

print(comparison.to_string(index=False))

print("\n========== INTERPRETATION ==========")

print("\nRandom Forest:")
print("Used for congestion-level classification.")
print(f"Validation accuracy: {random_forest_accuracy:.4f}")

print("\nXGBoost:")
print("Used for congestion-level classification.")
print(f"Validation accuracy: {xgboost_accuracy:.4f}")
print(f"Validation macro F1: {xgboost_macro_f1:.4f}")

print("\nLSTM:")
print("Used for next-step traffic-volume forecasting.")
print(f"MAE: {lstm_mae:.4f}")
print(f"RMSE: {lstm_rmse:.4f}")
print(f"R2: {lstm_r2:.4f}")

print("\n========== CONCLUSION ==========")

print(
    "XGBoost is the strongest classical model for congestion "
    "classification based on validation accuracy and macro F1."
)

print(
    "LSTM provides traffic-volume forecasting and is evaluated "
    "using regression metrics, so its metrics should not be "
    "directly compared with classification accuracy."
)

print("\nReport saved to:")
print("reports/model_comparison.csv")

print("\n==================================================")
print(" MODEL COMPARISON REPORT COMPLETED")
print("==================================================")