import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("==================================================")
print(" FLOWCAST LSTM MODEL EVALUATION")
print("==================================================")

# Load validation data
X_validation = np.load(
    "data/processed/lstm_X_validation_scaled.npy"
)

y_validation_scaled = np.load(
    "data/processed/lstm_y_validation_scaled.npy"
)

# Load trained LSTM model
model = load_model(
    "models/flowcast_lstm_best.keras"
)

# Load target scaler
target_scaler = joblib.load(
    "models/lstm_target_scaler.joblib"
)

print("\nValidation X shape:", X_validation.shape)
print("Validation y shape:", y_validation_scaled.shape)

# Make predictions
print("\n========== MAKING PREDICTIONS ==========")

y_pred_scaled = model.predict(
    X_validation,
    batch_size=128,
    verbose=1
)

# Convert predictions and actual values
# back to original traffic-volume scale
y_pred = target_scaler.inverse_transform(
    y_pred_scaled
).flatten()

y_actual = target_scaler.inverse_transform(
    y_validation_scaled.reshape(-1, 1)
).flatten()

# Calculate evaluation metrics
mae = mean_absolute_error(
    y_actual,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_actual,
        y_pred
    )
)

r2 = r2_score(
    y_actual,
    y_pred
)

print("\n========== LSTM EVALUATION RESULTS ==========")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# Show a few actual vs predicted values
print("\n========== SAMPLE PREDICTIONS ==========")

for i in range(10):
    print(
        f"Actual: {y_actual[i]:.2f} | "
        f"Predicted: {y_pred[i]:.2f}"
    )

# Save predictions
results = np.column_stack(
    (y_actual, y_pred)
)

np.savetxt(
    "data/processed/lstm_predictions.csv",
    results,
    delimiter=",",
    header="actual_traffic_volume,predicted_traffic_volume",
    comments=""
)

print("\nPredictions saved to:")
print("data/processed/lstm_predictions.csv")

print("\n==================================================")
print(" LSTM MODEL EVALUATION COMPLETED")
print("==================================================")