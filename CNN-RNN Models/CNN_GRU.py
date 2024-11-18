import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from tensorflow.keras import Sequential
from tensorflow.keras import metrics
import matplotlib.pyplot as plt
from Load_Data import load_data_for_model
from TimeHistory import TimeHistory

# Model Parameters:
batch_size = 16
epoch = 100
learning_rate = 0.00011831890118090147

# Load the preprocessed data
train_input, train_target, val_input, val_target, test_input, test_target = load_data_for_model()

# Define Model
model = Sequential()

# Define a normalization layer and fit on the training data
#normalizer = layers.Normalization(input_shape=(12,9))
#normalizer.adapt(train_input)

# Add Normalization Layer
#model.add(normalizer)

# CNN Layer (Feature Extraction)
model.add(layers.Conv1D(filters=56, kernel_size = 2, padding='valid', activation='relu', input_shape=(12,9)))

# Add Pooling Layer
model.add(layers.MaxPooling1D(pool_size=1, padding='valid'))

# GRU Layer (Temporal Memory)
model.add(layers.GRU(units=16, kernel_initializer='he_normal', return_sequences=True))
model.add(layers.GRU(units=56, kernel_initializer='he_normal', return_sequences=True))
model.add(layers.GRU(units=40, kernel_initializer='he_normal', return_sequences=False))

# Dense Layer
model.add(layers.Dense(64, activation='linear'))
model.add(layers.Dense(6, activation='linear'))
    # Final output layer will have 6 units (1 per each sensor to predict)
    # Uses linear activation for regression-based predictions

# Compile the Model
optimizer = optimizers.Adam(learning_rate=learning_rate)
    
model.compile(optimizer=optimizer, 
              loss='mae', 
              metrics=['mae', 'mse', metrics.RootMeanSquaredError(name='rmse'), metrics.R2Score(name="r2")])

# Define Callbacks for Early Stopping and Model Checkpoints
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

checkpoint = callbacks.ModelCheckpoint('Model_CNN-GRU.keras', monitor='val_loss', save_best_only=True, verbose=1)

time_callback = TimeHistory() 

# Print Model Summary
model.summary()

# Fit the Model (Start Training)

# Train the model
history = model.fit(
    train_input, train_target,
    epochs=epoch,  # Adjust epochs as needed
    batch_size=batch_size,  # Adjust batch size as needed
    validation_data=(val_input, val_target),
    callbacks=[early_stopping, checkpoint, time_callback],
    verbose=1  # Verbose output for training progress
)

# Evaluate the model on the test set
test_loss, test_mae, test_mse, test_rmse, test_r2= model.evaluate(test_input, test_target, verbose=1)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}, Test MSE: {test_mse}, Test RMSE: {test_rmse}, Test R²: {test_r2}")

# Plot the training loss and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Function over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()