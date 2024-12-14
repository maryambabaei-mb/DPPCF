import numpy as np
import pandas as pd

import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd

def replace_categorical_values(dataset, column, mapping):
    dataset[column] = dataset[column].replace(mapping)

def preprocess_compas(input_csv_path, output_csv_path):
    # Read the CSV file
    df = pd.read_csv(input_csv_path)

    # Define categorical columns
    categorical_columns = ['sex', 'race', 'c_charge_degree']

    # Create a mapping for categorical columns
    categorical_data_map = {}
    for col in categorical_columns:
        unique_values = df[col].unique()
        categorical_data_map[col] = {'values': unique_values}

    # Replace categorical values with numeric values
    for col, values_info in categorical_data_map.items():
        replace_categorical_values(df, col, dict(zip(values_info['values'], range(1, len(values_info['values']) + 1))))

    # Save the processed data to a new CSV file
    df.to_csv(output_csv_path, index=False)

    print("Processed data saved to", output_csv_path)

# Set the path to your original CSV file and the output CSV file
input_csv_path = '.\datasets\compas.csv'
output_csv_path = '.\datasets\processed_compas.csv'

# Preprocess the COMPAS dataset
preprocess_compas(input_csv_path, output_csv_path)