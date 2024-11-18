import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Imports
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
learning_rate = 0.0007240121373547832

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
model.add(layers.Conv1D(filters=48, kernel_size = 1, padding='valid', activation='relu', input_shape=(12,9)))
    # filters = 16 
    #   Number of filters recommended in the paper
    # kernel_size = 1
    #   Set to 1 which means CNN is applied across each feature independently making it suitable for time-series data
    # padding = 'valid'
    #   Valid padding which ensures no padding is applied to the input keeping the calculations efficient

model.add(layers.MaxPooling1D(pool_size=1, padding='valid'))
    # Pool Size: Pool Size of 1 reducing dimensions but keeping sequence intact

#model.add(layers.Conv1D(filters=32, kernel_size = 1, padding='valid', activation='relu'))
#model.add(layers.MaxPooling1D(pool_size=1, padding='valid'))

# LSTM Layer (Temporal Memory)
model.add(layers.LSTM(units=40, kernel_initializer='he_normal', return_sequences=False))
#model.add(layers.LSTM(units=32, kernel_initializer='he_normal', return_sequences=False))
    # units = 16 
    #   Number of units in the LSTM Layer
    # kernel_initializer = 'he_normal'
    #   Used for better weight initialization
    # return_sequences = False
    #   Ensures that LSTM layers return only the final hidden state which is typical when feeding output to a Dense Layer

# Dropout rate of 20% after LSTM
#model.add(layers.Dropout(0.2))  

# Dense Layer
model.add(layers.Dense(48, activation='linear'))
model.add(layers.Dense(6, activation='linear'))
    # Final output layer will have 6 units (1 per each sensor to predict)
    # Uses linear activation for regression-based predictions

# Compile the Model
optimizer = optimizers.Adam(learning_rate=learning_rate)
    # Optimizer minimizes the loss function
    # Adam (Adaptive Moment Estimation)
    
model.compile(optimizer=optimizer, 
              loss='mae', 
              metrics=['mae', 'mse', metrics.RootMeanSquaredError(name='rmse'), metrics.R2Score(name='r2')])
    # loss='mae'
    #   Loss Function is MAE (Mean Absolute Error)

# Define Callbacks for Early Stopping and Model Checkpoints
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # Stops Training Early when Model Stops Improving
    # monitor = 'val_loss': 
    #   Monitors validation loss to detect when model stops improving.
    # patience = 10: 
    #   If validation loss doesn't improve for 10 consecutive epochs, training will stop early.
    # restor_best_weights = True:
    #   Ensure model restores weights from the best epoch (lowest validation loss).
    
checkpoint = callbacks.ModelCheckpoint('Model_CNN-LSTM.keras', monitor='val_loss', save_best_only=True, verbose=1)
    # Save Model when Model Improves - this allows halting and continuing the training
    # monitor = 'val_loss':
    #   Checkpoint will save model when there is an improvement in validation loss.
    # save_best_only = True: 
    #   Saves only the best version of the model.
    # verbose = 1:
    #   Provides feedback when model is saved.

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

print("Complete")


