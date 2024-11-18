import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Array containing the file names
file_names = ['2021_GAS_DATASET_1.csv',
              '2021_GAS_DATASET_2.csv',
              '2022_GAS_DATASET_1.csv', 
              '2022_GAS_DATASET_2.csv', 
              '2022_GAS_DATASET_3.csv']  # Add more file names as needed

def process_dataset(file_path, sequence_length=12):
    """
    Processes a dataset to extract sequences of consecutive readings for neural network training.
    
    This function reads a dataset from a specified file, removes rows containing NaN values, 
    and constructs sequences of consecutive rows. Each sequence consists of `sequence_length` 
    rows as input, with the subsequent row serving as the target. Only sequences with 
    consecutive row indices are included to ensure data continuity.
    
    Parameters:
    ----------
    file_path : str
        Path to the CSV file containing the dataset.
    
    sequence_length : int, optional
        The length of the input sequence in rows (default is 12).
    
    Returns:
    -------
    input_data : np.ndarray
        A numpy array of shape (num_sequences, sequence_length, num_features), 
        where each entry is a valid sequence of `sequence_length` rows.
        
    targets : np.ndarray
        A numpy array of shape (num_sequences, num_features), containing the target row 
        for each sequence in `input_data`.
    
    Notes:
    ------
    - Rows containing NaN values are removed before processing, and the indices are 
      not reset to identify valid sequences with consecutive rows.
    - This function assumes the CSV file has a consistent structure without missing columns.

    """
    data = pd.read_csv(file_path)
    
    # Drop rows with NaN values, but do not reset the index
    data = data.dropna()
    data_array = data.values
    indices = data.index  # Get the indices after dropping NaNs
    num_samples = len(data_array) - sequence_length
    
    # Initialize lists for storing input sequences and targets
    input_data = []
    targets = []
    
    # Iterate over possible sequences
    for i in range(num_samples):
        # Extract the sequence of indices we want to check
        index_sequence = indices[i:i + sequence_length + 1]
        
        # Check if indices are consecutive
        if np.all(np.diff(index_sequence) == 1):
            input_data.append(data_array[i:i + sequence_length])   # 12 entries for input
            targets.append(data_array[i + sequence_length][1:])    # 13th entry as target
    
    # Convert lists to numpy arrays
    input_data = np.array(input_data)
    targets = np.array(targets)
    
    return input_data, targets

# Set the current working directory to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Initialize empty lists to collect data
all_input_data = []
all_targets = []

# Process each dataset and combine the results
for file_name in file_names:
    input_data, targets = process_dataset(file_name)
    all_input_data.append(input_data)
    all_targets.append(targets)

# Combine all input data and targets from all datasets
combined_input_data = np.vstack(all_input_data)
combined_targets = np.vstack(all_targets)

# Randomly split the combined data into training (80%), validation (10%), and test (10%)
train_input, temp_input, train_target, temp_target = train_test_split(
    combined_input_data, combined_targets, test_size=0.2, random_state=42
)

val_input, test_input, val_target, test_target = train_test_split(
    temp_input, temp_target, test_size=0.5, random_state=42
)

# Save the data splits for reproducibility
np.save('train_input.npy', train_input)
np.save('train_target.npy', train_target)
np.save('val_input.npy', val_input)
np.save('val_target.npy', val_target)
np.save('test_input.npy', test_input)
np.save('test_target.npy', test_target)

print("Data splits saved successfully.")
print(f"Training data shape: {train_input.shape}")
print(f"Validation data shape: {val_input.shape}")
print(f"Test data shape: {test_input.shape}")