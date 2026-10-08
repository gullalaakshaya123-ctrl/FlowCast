import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler

print("==================================================")
print(" FLOWCAST LSTM DATA SCALING")
print("==================================================")

# --------------------------------------------------
# 1. Load train and validation data
# --------------------------------------------------

X_train = np.load(
    "data/processed/lstm_X_train.npy"
)

y_train = np.load(
    "data/processed/lstm_y_train.npy"
)

X_validation = np.load(
    "data/processed/lstm_X_validation.npy"
)

y_validation = np.load(
    "data/processed/lstm_y_validation.npy"
)

print("\nOriginal training X shape:", X_train.shape)
print("Original validation X shape:", X_validation.shape)

# --------------------------------------------------
# 2. Create scalers
# --------------------------------------------------

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# --------------------------------------------------
# 3. Fit feature scaler ONLY on training data
# --------------------------------------------------

# X shape:
# (samples, time_steps, features)

n_train_samples = X_train.shape[0]
n_time_steps = X_train.shape[1]
n_features = X_train.shape[2]

# Reshape to 2D for sklearn scaler
X_train_2d = X_train.reshape(
    -1,
    n_features
)

X_validation_2d = X_validation.reshape(
    -1,
    n_features
)

# Fit only on training data
feature_scaler.fit(X_train_2d)

# Transform train and validation
X_train_scaled_2d = feature_scaler.transform(
    X_train_2d
)

X_validation_scaled_2d = feature_scaler.transform(
    X_validation_2d
)

# Reshape back to LSTM format
X_train_scaled = X_train_scaled_2d.reshape(
    n_train_samples,
    n_time_steps,
    n_features
)

X_validation_scaled = X_validation_scaled_2d.reshape(
    X_validation.shape[0],
    X_validation.shape[1],
    X_validation.shape[2]
)

# --------------------------------------------------
# 4. Scale target
# --------------------------------------------------

y_train_2d = y_train.reshape(-1, 1)
y_validation_2d = y_validation.reshape(-1, 1)

# Fit target scaler ONLY on training target
target_scaler.fit(y_train_2d)

y_train_scaled = target_scaler.transform(
    y_train_2d
).flatten()

y_validation_scaled = target_scaler.transform(
    y_validation_2d
).flatten()

# --------------------------------------------------
# 5. Display results
# --------------------------------------------------

print("\n========== SCALED DATA ==========")

print(
    "Training X shape:",
    X_train_scaled.shape
)

print(
    "Validation X shape:",
    X_validation_scaled.shape
)

print(
    "Training y shape:",
    y_train_scaled.shape
)

print(
    "Validation y shape:",
    y_validation_scaled.shape
)

print("\nTraining X range:")
print(
    "Minimum:",
    X_train_scaled.min()
)

print(
    "Maximum:",
    X_train_scaled.max()
)

print("\nTraining y range:")
print(
    "Minimum:",
    y_train_scaled.min()
)

print(
    "Maximum:",
    y_train_scaled.max()
)

# --------------------------------------------------
# 6. Save scaled data
# --------------------------------------------------

np.save(
    "data/processed/lstm_X_train_scaled.npy",
    X_train_scaled
)

np.save(
    "data/processed/lstm_y_train_scaled.npy",
    y_train_scaled
)

np.save(
    "data/processed/lstm_X_validation_scaled.npy",
    X_validation_scaled
)

np.save(
    "data/processed/lstm_y_validation_scaled.npy",
    y_validation_scaled
)

# --------------------------------------------------
# 7. Save scalers
# --------------------------------------------------

joblib.dump(
    feature_scaler,
    "models/lstm_feature_scaler.joblib"
)

joblib.dump(
    target_scaler,
    "models/lstm_target_scaler.joblib"
)

print("\n========== FILES SAVED ==========")

print("data/processed/lstm_X_train_scaled.npy")
print("data/processed/lstm_y_train_scaled.npy")
print("data/processed/lstm_X_validation_scaled.npy")
print("data/processed/lstm_y_validation_scaled.npy")

print("\nmodels/lstm_feature_scaler.joblib")
print("models/lstm_target_scaler.joblib")

print("\n==================================================")
print(" LSTM DATA SCALING COMPLETED")
print("==================================================")