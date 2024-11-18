import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Array containing the file names
file_names = ['GAS_DATASET_1.csv', 'GAS_DATASET_2.csv', 'GAS_DATASET_3.csv']  # Add more file names as needed

def process_dataset(file_path, sequence_length=12, num_segments=10):
    """
    Processes a dataset by dividing it into segments, splitting each segment into training, validation, 
    and testing portions, and then extracting valid sequences of consecutive readings.
    
    Parameters:
    ----------
    file_path : str
        Path to the CSV file containing the dataset.
    
    sequence_length : int, optional
        The length of the input sequence in rows (default is 12).
        
    num_segments : int, optional
        The number of segments to divide the dataset into (default is 10).
    
    Returns:
    -------
    train_data, val_data, test_data : tuples of np.ndarray
        Arrays of input sequences and target values for training, validation, and testing.
    """
    data = pd.read_csv(file_path)
    
    # Drop rows with NaN values, but do not reset the index
    data = data.dropna()
    segment_size = len(data) // num_segments
    
    train_data = []
    val_data = []
    test_data = []
    
    # Process each segment individually
    for i in range(num_segments):
        segment_start = i * segment_size
        segment_end = (i + 1) * segment_size if i < num_segments - 1 else len(data)
        segment = data.iloc[segment_start:segment_end]
        segment_indices = segment.index
        
        # Define split points: last 10% for testing, previous 10% for validation
        test_start = int(0.9 * len(segment))
        val_start = int(0.8 * len(segment))
        
        train_segment = segment.iloc[:val_start]
        val_segment = segment.iloc[val_start:test_start]
        test_segment = segment.iloc[test_start:]
        
        # Function to extract valid sequences from a given segment
        def extract_sequences(data_segment):
            inputs, targets = [], []
            for j in range(len(data_segment) - sequence_length):
                index_sequence = data_segment.index[j:j + sequence_length + 1]
                
                # Check if indices are consecutive
                if np.all(np.diff(index_sequence) == 1):
                    # Include all columns for input sequences
                    input_seq = data_segment.iloc[j:j + sequence_length, :].values  # All columns
                    # Select only pollutant columns for targets
                    target = data_segment.iloc[j + sequence_length][['PM25', 'PM10', 'O3', 'NO2', 'SO2', 'CO']].values
                    inputs.append(input_seq)
                    targets.append(target)
            return np.array(inputs), np.array(targets)
        
        # Extract sequences from each set within the segment
        train_inputs, train_targets = extract_sequences(train_segment)
        val_inputs, val_targets = extract_sequences(val_segment)
        test_inputs, test_targets = extract_sequences(test_segment)
        
        # Append to main lists
        train_data.append((train_inputs, train_targets))
        val_data.append((val_inputs, val_targets))
        test_data.append((test_inputs, test_targets))
    
    return train_data, val_data, test_data

# Set current working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Initialize empty lists to collect data
all_train_data = []
all_val_data = []
all_test_data = []

# Process each dataset and combine the results
for file_name in file_names:
    train, val, test = process_dataset(file_name)
    all_train_data.extend(train)
    all_val_data.extend(val)
    all_test_data.extend(test)

# Combine all input data and targets from all datasets
combined_train_input = np.vstack([x[0] for x in all_train_data])
combined_train_target = np.vstack([x[1] for x in all_train_data])

combined_val_input = np.vstack([x[0] for x in all_val_data])
combined_val_target = np.vstack([x[1] for x in all_val_data])

combined_test_input = np.vstack([x[0] for x in all_test_data])
combined_test_target = np.vstack([x[1] for x in all_test_data])

# Standardize based on training data only
scaler = StandardScaler()
num_features = combined_train_input.shape[-1]  # number of features in each timestep

# Reshape input to 2D for scaling: (total_samples * timesteps, num_features)
train_input_reshaped = combined_train_input.reshape(-1, num_features)
scaler.fit(train_input_reshaped)  # Fit on training data only

# Save the scaler for future use
joblib.dump(scaler, 'scaler.joblib')

# Apply the scaler to transform the training, validation, and test data
train_input_scaled = scaler.transform(train_input_reshaped).reshape(combined_train_input.shape)
val_input_scaled = scaler.transform(combined_val_input.reshape(-1, num_features)).reshape(combined_val_input.shape)
test_input_scaled = scaler.transform(combined_test_input.reshape(-1, num_features)).reshape(combined_test_input.shape)

# Save the data splits
np.save('train_input.npy', train_input_scaled)
np.save('train_target.npy', combined_train_target)
np.save('val_input.npy', val_input_scaled)
np.save('val_target.npy', combined_val_target)
np.save('test_input.npy', test_input_scaled)
np.save('test_target.npy', combined_test_target)

print("Data splits and scaler saved successfully.")
print(f"Training data shape: {train_input_scaled.shape}")
print(f"Training target shape: {combined_train_target.shape}")
print(f"Validation data shape: {val_input_scaled.shape}")
print(f"Validation target shape: {combined_val_target.shape}")
print(f"Test data shape: {test_input_scaled.shape}")
print(f"Test target shape: {combined_test_target.shape}")
