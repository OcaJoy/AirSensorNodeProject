import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from KalmanFilter import KalmanFilter  # Assuming the KalmanFilter class is imported

#####################################################################################
### LOAD DATA ####
##################

def csv_to_dataframe(csv_file):
    """
    Read the CSV file into a pandas DataFrame and return it.
    """
    return pd.read_csv(csv_file)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define file path relative to the script's directory
test_data_noisy_file = os.path.join(script_dir, 'TestDataNoisy.csv')

# Load the data into a DataFrame
data = csv_to_dataframe(test_data_noisy_file)

# Separate original and noisy columns
original_columns = [col for col in data.columns if not col.endswith('_Noisy')]
noisy_columns = [col for col in data.columns if col.endswith('_Noisy')]

# Extract original and noisy data
original_data = data[original_columns].to_numpy()
noisy_data = data[noisy_columns].to_numpy()

# Create a time array starting from 0, incrementing by 5 seconds for each time step
time_steps = np.arange(0, original_data.shape[0] * 5, 5)

#####################################################################################
### SET UP VARIANCES ####
#########################

# Variances for each sensor
variances = {
    "O3": 1.5,
    "NO2": 0.8,
    "Temperature": 1.12,
    "Humidity": 0.75
}

# Construct the R matrix using the specified variances
R = np.diag([variances[col] for col in original_columns if col in variances])

#####################################################################################
### APPLY KALMAN FILTER ####
############################

# Initialize Kalman Filter with the first noisy reading
kf = KalmanFilter(
    sensor_dim=noisy_data.shape[1],  # Number of sensors
    Q=0.0001,  # Process noise covariance
    R=R,  # Measurement noise covariance
    initial_reading=noisy_data[0]  # Initial state estimate
)

# Apply Kalman filter to each noisy measurement
filtered_output = []

for noisy_measurement in noisy_data:
    filtered_measurement = kf.filter(noisy_measurement)
    filtered_output.append(filtered_measurement)

# Convert filtered_output to numpy array for easier manipulation
filtered_output = np.array(filtered_output)

#####################################################################################
### PLOT RESULTS ####
#####################

plt.figure(figsize=(12, 8))

# Iterate through available sensors and plot
for i, sensor in enumerate(original_columns):
    original_data_col = original_data[:, i]
    noisy_data_col = noisy_data[:, i]
    filtered_data_col = filtered_output[:, i]

    plt.subplot(2, 2, i + 1)  # 2 rows and 2 columns for the 4 sensors
    plt.plot(time_steps, original_data_col, label='Original Data', color='green', alpha=0.5)
    plt.plot(time_steps, noisy_data_col, label='Noisy Data', color='red', alpha=0.5)
    plt.plot(time_steps, filtered_data_col, label='Filtered Output', color='blue')

    plt.title(sensor)
    plt.xlabel('Time (seconds)')
    plt.ylabel(sensor)
    plt.legend()

plt.tight_layout()
plt.show()
