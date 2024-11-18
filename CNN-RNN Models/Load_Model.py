import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from keras import models
from custom_layers import iLSTM
import joblib

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load the Model
model = models.load_model('Model_CNN-LSTM.keras') # Load CNN-LSTM
#model = models.load_model('Model_CNN-GRU.keras') # Load CNN-GRU
#model = models.load_model('Model_CNN-iLSTM.keras', custom_objects={'iLSTM':iLSTM}) # Load CNN-iLSTM

# Load scaler
scaler = joblib.load('Data/scaler.joblib')

# Provided data for the input sequence (12 time steps, 9 features)
input_data = np.array([
    [23, 7, 15, 22, 3, 0.2, 0.19, -3.7, 69],
    [0, 5, 13, 25, 2, 0.2, 0.18, -3, 68],
    [1, 5, 11, 28, 2, 0.2, 0.17, -2.2, 80],
    [2, 4, 8, 31, 1, 0.1, 0.15, -1.2, 85],
    [3, 4, 9, 30, 1, 0.1, 0.15, -0.6, 86],
    [4, 5, 11, 30, 1, 0.1, 0.15, -0.1, 87],
    [5, 5, 11, 30, 1, 0.1, 0.15, 0.8, 87],
    [6, 7, 13, 30, 1, 0.1, 0.16, 1.6, 86],
    [7, 6, 10, 30, 1, 0.2, 0.16, 2.4, 91],
    [8, 6, 11, 30, 1, 0.2, 0.15, 3.3, 91],
    [9, 8, 15, 32, 1, 0.2, 0.16, 3.9, 92],
    [10, 11, 18, 34, 2, 0.2, 0.17, 4, 95]
])

# Expected output for comparison
expected_output = np.array([11, 18, 36, 2, 0.2, 0.16])

# Reshape input data to 2D for scaling: (total_samples * timesteps, num_features)
# This makes the shape (12, 9), which is compatible with scaler
input_data_scaled = scaler.transform(input_data.reshape(-1, input_data.shape[-1]))

# Reshape scaled data back to 3D for model input: (batch_size, time_steps, features)
input_data_scaled = input_data_scaled.reshape((1, 12, 9))  # 1 batch, 12 time steps, 9 features

# Make a prediction
#predicted_output_standardized = model.predict(input_data_standardized)
predicted_output = model.predict(input_data_scaled)


# Print results
print("Predicted Output:", predicted_output[0])
print("Expected Output:", expected_output)

