import numpy as np
import os

def load_data_for_model():
    """
    Loads the preprocessed data from .npy files created by DataSetUp.py.
    
    Returns:
        train_input (np.ndarray): Training input data.
        train_target (np.ndarray): Training target data.
        val_input (np.ndarray): Validation input data.
        val_target (np.ndarray): Validation target data.
        test_input (np.ndarray): Test input data.
        test_target (np.ndarray): Test target data.
    """
    
    # Set the current working directory to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Load the data splits from .npy files
    train_input = np.load('Data/train_input.npy')
    train_target = np.load('Data/train_target.npy')
    val_input = np.load('Data/val_input.npy')
    val_target = np.load('Data/val_target.npy')
    test_input = np.load('Data/test_input.npy')
    test_target = np.load('Data/test_target.npy')

    # Print shapes for confirmation
    print("Data loaded successfully:")
    print(f"Training data shape: {train_input.shape}")
    print(f"Training target shape: {train_target.shape}")
    print(f"Validation data shape: {val_input.shape}")
    print(f"Validation target shape: {val_target.shape}")
    print(f"Test data shape: {test_input.shape}")
    print(f"Test target shape: {test_target.shape}")

    return train_input, train_target, val_input, val_target, test_input, test_target
