import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from KalmanFilter import KalmanFilter

display_value = False

#####################################################################################
### LOAD DATA ####
##################

def csv_to_array(csv_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Extract column headers (sensor names)
    headers = df.columns.tolist()

    # Convert the DataFrame to a numpy array (numeric data only)
    data = df.to_numpy(dtype=float)

    # Return the headers separately from the numeric data
    return headers, data

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define file path relative to the script's directory
test_data_noisy_file = os.path.join(script_dir, 'TestDataNoisy.csv')

# Convert CSV file to array (headers, numeric data)
headers, test_data_noisy = csv_to_array(test_data_noisy_file)

# Extract sensor names and corresponding noisy sensor names
sensors = headers[:len(headers) // 2]  # Original sensor names
noisy_sensors = headers[len(headers) // 2:]  # Corresponding noisy sensor names

# Create a time array starting from 0, incrementing by 5 seconds for each time step
time_steps = np.arange(0, test_data_noisy.shape[0] * 5, 5)

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
R = np.diag([variances[sensor] for sensor in sensors if sensor in variances])

if display_value:
    # Output the R matrix
    print("Variance of each sensor (R matrix diagonal elements):")
    print([variances[sensor] for sensor in sensors if sensor in variances])

    print("\nR matrix (Measurement noise covariance matrix):")
    print(R)

#####################################################################################
### APPLY KALMAN FILTER ####
############################

# Extract noisy data for Kalman filtering
noisy_data = test_data_noisy[:, len(sensors):]  # Assuming noisy readings are in second half

# Initialize Kalman Filter with the first noisy reading
kf = KalmanFilter(
    sensor_dim=4,
    Q=0.001,  # Process noise covariance
    R=R,
    initial_reading=noisy_data[0]
)

# Apply Kalman filter to each noisy measurement
filtered_output = []

for noisy_measurement in noisy_data:
    filtered_measurement = kf.filter(noisy_measurement)
    filtered_output.append(filtered_measurement)

# Convert filtered_output to numpy array for easier manipulation
filtered_output = np.array(filtered_output)

if display_value:
    # Output the filtered results
    print("Filtered Output:")
    print(filtered_output)

#####################################################################################
### PLOT RESULTS ####
#####################

plt.figure(figsize=(12, 8))

for i, sensor in enumerate(sensors):
    if sensor in variances:  # Only plot sensors that are in the variance dictionary
        plt.subplot(2, 2, i + 1)
        original_data = test_data_noisy[:, i]
        noisy_data_col = test_data_noisy[:, len(sensors) + i]
        filtered_data_col = filtered_output[:, i]

        plt.plot(time_steps, original_data, label='Original Data', color='green', alpha=0.5)
        plt.plot(time_steps, noisy_data_col, label='Noisy Data', color='red', alpha=0.5)
        plt.plot(time_steps, filtered_data_col, label='Filtered Output', color='blue')

        plt.title(sensor)
        plt.xlabel('Time (seconds)')
        plt.ylabel(sensor)
        plt.legend()

plt.tight_layout()
plt.show()
