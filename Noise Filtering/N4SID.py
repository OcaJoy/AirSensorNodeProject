import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nfoursid.kalman import Kalman
from nfoursid.nfoursid import NFourSID
from nfoursid.state_space import StateSpace

display_value = False

#########################################################################################
### LOAD DATA ###
#################
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define file paths relative to the script's directory
sensor_data_file = os.path.join(script_dir, 'sensor_data.csv')
noisy_data_file = os.path.join(script_dir, 'noisy_data.csv')

# Convert both CSV files to arrays (headers, numeric data)
sensor_data = pd.read_csv('sensor_data.csv')
noisy_data = pd.read_csv('noisy_data.csv')

#########################################################################################
### N4SID ###
#############
# Number of Block Hankel Matricess
num_block_rows = 2
    # Specifies how many past measurements and inputs are used to form the block Hankel
    # matrices - fundamental to subpsace identification in N4SID.
    # Start with choosing small values and experiment by increasing it. 

# Output Columns
output_columns = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO', 'CO2', 'Temperature', 'Humidity']

# Initialize N4SID
n4sid_noisy = NFourSID(noisy_data, output_columns=output_columns, num_block_rows=num_block_rows)

# Perform subspace identification
n4sid_noisy.subspace_identification()

# Plot Eigen Values
fig, ax = plt.subplots()
n4sid_noisy.plot_eigenvalues(ax)
fig.tight_layout()
plt.show()
    
# Perform System Identification
system_noisy, cov_noisy = n4sid_noisy.system_identification()

# Initialize Kalman Filter 
kalman_filter = Kalman(state_space=system_noisy, noise_covariance=cov_noisy)

# Prepare to store results
filtered_results = []

# Iterate over each data point in your noisy data
for index in range(len(noisy_data)):
    # Get the current observed output (noisy measurement) and reshape it to (9, 1)
    y_observed = noisy_data.iloc[index].values.reshape(-1, 1)

    # Create a zero input vector of the right shape
    u = np.zeros((0, 1))  # This creates a zero vector for the input
    
    # Step the Kalman filter with the observed output and the input (None)
    kalman_filter.step(y=y_observed, u=u)

# Now you can retrieve the filtered results
filtered_results = kalman_filter.to_dataframe()

# Extract the PM2.5 values from each DataFrame
pm25_original = sensor_data['PM2.5'].values
pm25_noisy = noisy_data['PM2.5'].values

# Access the filtered PM2.5 values using multi-indexing
pm25_filtered = filtered_results[('$y_0$', 'filtered', 'output')].values  # Adjust this if $y_0$ does not correspond to PM2.5

filtered_results.to_csv("filtered_results.csv", index=False)  # index=False prevents adding the row number column

# Create a time axis assuming equal spacing
time = np.arange(len(pm25_filtered))

# Plot the values
plt.figure(figsize=(12, 6))
plt.plot(time, pm25_original, label='Original PM2.5', color='blue', alpha=0.6)
plt.plot(time, pm25_noisy, label='Noisy PM2.5', color='red', alpha=0.6)
plt.plot(time, pm25_filtered, label='Filtered PM2.5', color='green', linewidth=2)

# Adding labels and title
plt.xlabel('Time (arbitrary units)')
plt.ylabel('PM2.5 Concentration')
plt.title('PM2.5 Measurements Comparison')
plt.legend()
plt.grid()

# Show the plot
plt.show()

if(display_value):
    # Extract the A, B, C, D matrices for the noisy data
    A_noisy, B_noisy, C_noisy, D_noisy = system_noisy.a, system_noisy.b, system_noisy.c, system_noisy.d
    # Display results
    print("System Matrices for Noisy Data:")
    print("A:\n", A_noisy)
    print("B:\n", B_noisy)
    print("C:\n", C_noisy)
    print("D:\n", D_noisy)
    print("Covariance:\n", cov_noisy)
