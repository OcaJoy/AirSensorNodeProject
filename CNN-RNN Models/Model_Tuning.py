"""
Keras Tuner:
@misc{omalley2019kerastuner,
    title        = {KerasTuner},
    author       = {O'Malley, Tom and Bursztein, Elie and Long, James and Chollet, Francois and Jin, Haifeng and Invernizzi, Luca and others},
    year         = 2019,
    howpublished = https://github.com/keras-team/keras-tuner
}

This script is used to find the best hyperparameters for the CNN-LSTM, CNN-GRU, and CNN-iLSTM Model. 
"""

import os
import keras_tuner as kt
from tensorflow.keras import layers, callbacks, optimizers, Sequential, metrics
import matplotlib.pyplot as plt
from Load_Data import load_data_for_model
from TimeHistory import TimeHistory
from custom_layers import iLSTM  # Assuming you have a custom iLSTM layer

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load preprocessed data
train_input, train_target, val_input, val_target, test_input, test_target = load_data_for_model()

# Hyperparameter configuration
HYPERPARAMS = {
    "num_cnn_layers": (1, 3),             # Number of CNN layers: min 1, max 3
    "filters": (16, 64, 8),               # Filters for CNN: min 16, max 64, step 8
    "kernel_size": [1, 2, 3],             # Kernel sizes for CNN layers
    
    "num_rnn_layers": (1, 3),             # Number of RNN layers: min 1, max 3
    "units_rnn": (16, 64, 8),             # Units in RNN: min 16, max 64, step 8

    "dense_units": (16, 128, 16),         # Units for 2nd-to-last dense layer: min 16, max 128, step 16

    "learning_rate": (1e-4, 1e-2),        # Learning rate: min 1e-4, max 1e-2 (log scale)
}

# Choose the model type to tune: 'CNN-LSTM', 'CNN-GRU', or 'CNN-iLSTM'
#model_type = 'CNN-LSTM'  # Tune for CNN-LSTM Model
#model_type = 'CNN-GRU'  # Tune for CNN-GRU Model
model_type = 'CNN-iLSTM'  # Tune for CNN-iLSTM Model

# Define the model-building function with Keras Tuner
def build_model(hp, model_type='CNN-LSTM'):
    model = Sequential()

    # Add CNN layers
    for i in range(hp.Int('num_cnn_layers', *HYPERPARAMS["num_cnn_layers"])):  # Tune the number of CNN layers
        filters = hp.Int(f'filters_{i}', min_value=HYPERPARAMS["filters"][0], 
                         max_value=HYPERPARAMS["filters"][1], step=HYPERPARAMS["filters"][2])
        kernel_size = hp.Choice(f'kernel_size_{i}', values=HYPERPARAMS["kernel_size"])
        
        model.add(layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='valid',
            activation='relu',
            input_shape=(12, 9) if i == 0 else None  # Only set input shape for the first layer
        ))
        model.add(layers.MaxPooling1D(pool_size=1, padding='valid'))

    # Add RNN layers (LSTM, GRU, or iLSTM based on model_type)
    num_rnn_layers = hp.Int('num_rnn_layers', *HYPERPARAMS["num_rnn_layers"])  # Tune the number of RNN layers
    for j in range(num_rnn_layers):
        units = hp.Int(f'units_{j}', min_value=HYPERPARAMS["units_rnn"][0], 
                       max_value=HYPERPARAMS["units_rnn"][1], step=HYPERPARAMS["units_rnn"][2])
        
        # Set `return_sequences=False` for the last RNN layer
        return_sequences = (j < num_rnn_layers - 1)  # Only True for RNN layers before the last one

        # Define the RNN layer without regularization or dropout
        if model_type == 'CNN-LSTM':
            layer = layers.LSTM(units=units, kernel_initializer='he_normal', return_sequences=return_sequences)
        elif model_type == 'CNN-GRU':
            layer = layers.GRU(units=units, kernel_initializer='he_normal', return_sequences=return_sequences)
        elif model_type == 'CNN-iLSTM':
            layer = layers.RNN(iLSTM(units=units), return_sequences=return_sequences)
        
        model.add(layer)

    # Optional dense layer before the output layer
    dense_units = hp.Int('dense_units', min_value=HYPERPARAMS["dense_units"][0], 
                         max_value=HYPERPARAMS["dense_units"][1], step=HYPERPARAMS["dense_units"][2])
    model.add(layers.Dense(units=dense_units, activation='relu'))
    
    # Output layer
    model.add(layers.Dense(6, activation='linear'))

    # Compile the model with a tunable learning rate
    learning_rate = hp.Float('learning_rate', min_value=HYPERPARAMS["learning_rate"][0], max_value=HYPERPARAMS["learning_rate"][1], sampling='log')
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mae',
        metrics=['mae', 'mse', metrics.RootMeanSquaredError(name='rmse'), metrics.R2Score(name="r2")]
    )

    return model

# Main function to set up and run Keras Tuner based on selected model type
def run_tuner(model_type):
    tuner = kt.Hyperband(
        lambda hp: build_model(hp, model_type=model_type),
        objective='val_loss',
        max_epochs=80,  # Set max epochs for each tuning trial
        factor=3,
        directory='tuning_dir',
        project_name=f'tune_{model_type}'
    )

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Run the tuner search
    tuner.search(
        train_input, train_target,
        epochs=80,
        validation_data=(val_input, val_target),
        callbacks=[early_stopping]
    )

    # Print a summary of the tuning results
    tuner.results_summary()
    
    # Get the best hyperparameters and model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]  # Using `num_models` instead of `num_trials`
    
    # Print the best hyperparameters
    print(f"Best hyperparameters for {model_type}:")
    for param in best_hps.values:
        print(f"{param}: {best_hps.get(param)}")

    # Evaluate the best model on the test data
    test_loss, test_mae, test_mse, test_rmse, test_r2 = best_model.evaluate(test_input, test_target, verbose=1)
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}, Test MSE: {test_mse}, Test RMSE: {test_rmse}, Test R²: {test_r2}")

# Run Tuner:
run_tuner(model_type)
