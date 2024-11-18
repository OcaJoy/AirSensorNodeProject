import pandas as pd
import numpy as np
import os

# Set the current working directory to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Set file name to save as:
file_name = 'GAS_DATASET.csv'

# File names with absolute paths
files = {
    'PM25': os.path.join(script_dir, 'PM25_USE.csv'),
    'PM10': os.path.join(script_dir, 'PM10_USE.csv'),
    'O3': os.path.join(script_dir, 'O3_USE.csv'),
    'NO2': os.path.join(script_dir, 'NO2_USE.csv'),
    'SO2': os.path.join(script_dir, 'SO2_USE.csv'),
    'CO': os.path.join(script_dir, 'CO_USE.csv'),
    'Temperature_Humidity': os.path.join(script_dir, 'TEMP_HUM_USE.csv')
}

# Function to read and transpose a gas file
def read_and_transpose(file_path):
    df = pd.read_csv(file_path)
    df_transposed = df.set_index(df.columns[0]).stack().reset_index()
    df_transposed.columns = ['Date', 'Hour', 'Value']
    return df_transposed['Value']

# Reading and combining gas sensor data
combined_data = pd.DataFrame()

for col_name, file_path in files.items():
    if col_name != 'Temperature_Humidity':
        if os.path.exists(file_path):
            col_data = read_and_transpose(file_path)
            combined_data[col_name] = col_data
        else:
            print(f"File not found: {file_path}")
            continue

# Reading temperature and humidity data
if os.path.exists(files['Temperature_Humidity']):
    temp_hum_df = pd.read_csv(files['Temperature_Humidity'])
    # Assign Temperature and Humidity, ensuring they’re added as numeric types
    combined_data['Temperature'] = pd.to_numeric(temp_hum_df.iloc[:, 1], errors='coerce')
    combined_data['Humidity'] = pd.to_numeric(temp_hum_df.iloc[:, 2], errors='coerce')
else:
    print(f"File not found: {files['Temperature_Humidity']}")

# Add the hour column
combined_data.insert(0, 'Hour', combined_data.index % 24)

# Replace -999 and empty cells (or single spaces) with NaN
combined_data.replace([-999, '', ' '], np.nan, inplace=True)

# Save the combined data to a CSV file with limited precision
output_file = os.path.join(script_dir, file_name)
combined_data.to_csv(output_file, index=False, float_format='%.2f')

print(f"Combined data saved to {output_file}")

# Perform conditional interpolation for each column
for column in combined_data.columns:
    if column == 'Hour':  # Skip the hour column as it does not contain NaN values
        continue
    
    # Identify segments of single NaNs
    na_segments = combined_data[column].isna().astype(int).groupby(combined_data[column].notna().cumsum()).cumsum()
    
    # Process only single NaN values with both preceding and following valid values
    single_na_mask = (na_segments == 1)
    for idx in combined_data[single_na_mask].index:
        # Check if both preceding and following values exist and are valid
        if idx - 1 >= 0 and idx + 1 < len(combined_data) and not pd.isna(combined_data.loc[idx - 1, column]) and not pd.isna(combined_data.loc[idx + 1, column]):
            # Create a temporary subset for interpolation
            temp_series = combined_data[column].loc[idx - 1:idx + 1]
            interpolated_value = temp_series.interpolate(method='linear').loc[idx]
            combined_data.loc[idx, column] = interpolated_value

# Save the updated combined data to a new CSV file
output_file_interpolated = os.path.join(script_dir, 'GAS_DATASET_INTERPOLATED.csv')
combined_data.to_csv(output_file_interpolated, index=False, float_format='%.2f')

print(f"Interpolated data saved to {output_file_interpolated}")

