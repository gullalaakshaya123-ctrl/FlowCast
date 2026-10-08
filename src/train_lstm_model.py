import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

print("==================================================")
print(" FLOWCAST LSTM MODEL TRAINING")
print("==================================================")

# --------------------------------------------------
# 1. Load scaled training and validation data
# --------------------------------------------------

X_train = np.load(
    "data/processed/lstm_X_train_scaled.npy"
)

y_train = np.load(
    "data/processed/lstm_y_train_scaled.npy"
)

X_validation = np.load(
    "data/processed/lstm_X_validation_scaled.npy"
)

y_validation = np.load(
    "data/processed/lstm_y_validation_scaled.npy"
)

print("\nTraining X shape:", X_train.shape)
print("Training y shape:", y_train.shape)

print("Validation X shape:", X_validation.shape)
print("Validation y shape:", y_validation.shape)

# --------------------------------------------------
# 2. Build LSTM model
# --------------------------------------------------

model = Sequential([
    LSTM(
        64,
        return_sequences=True,
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),

    Dropout(0.2),

    LSTM(32),

    Dropout(0.2),

    Dense(16, activation="relu"),

    Dense(1)
])

# --------------------------------------------------
# 3. Compile model
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

print("\n========== MODEL SUMMARY ==========")

model.summary()

# --------------------------------------------------
# 4. Callbacks
# --------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    "models/flowcast_lstm_best.keras",
    monitor="val_loss",
    save_best_only=True
)

# --------------------------------------------------
# 5. Train model
# --------------------------------------------------

print("\n========== TRAINING STARTED ==========")

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_validation,
        y_validation
    ),

    epochs=30,
    batch_size=128,

    callbacks=[
        early_stopping,
        model_checkpoint
    ],

    verbose=1
)

# --------------------------------------------------
# 6. Save final model
# --------------------------------------------------

model.save(
    "models/flowcast_lstm_final.keras"
)

print("\n========== TRAINING COMPLETED ==========")

print(
    "Epochs completed:",
    len(history.history["loss"])
)

print(
    "Best validation loss:",
    min(history.history["val_loss"])
)

print(
    "Best validation MAE:",
    min(history.history["val_mae"])
)

print("\nSaved models:")

print(
    "models/flowcast_lstm_best.keras"
)

print(
    "models/flowcast_lstm_final.keras"
)

print("\n==================================================")
print(" FLOWCAST LSTM MODEL TRAINING COMPLETED")
print("==================================================")