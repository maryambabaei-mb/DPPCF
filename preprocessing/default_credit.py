import pandas as pd

def replace_categorical_values(dataset, column, mapping):
    dataset[column] = dataset[column].replace(mapping)

def preprocess_default_credit(input_csv_path, output_csv_path):
    # Read the CSV file
    df = pd.read_csv(input_csv_path)

    # Define categorical columns
    categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    # Create a mapping for categorical columns
    categorical_data_map = {
        'SEX': {'Male': 1, 'Female': 2},
        'EDUCATION': {'Graduate_school': 1, 'University': 2, 'High_school': 3, 'Others': 4},
        'MARRIAGE': {'Married': 1, 'Single': 2, 'Others': 3},
        'PAY_0': {'Pay_duly': 0, 'Pay_delay>=1': 1},
        'PAY_2': {'Pay_duly': 0, 'Pay_delay>=1': 1},
        'PAY_3': {'Pay_duly': 0, 'Pay_delay>=1': 1},
        'PAY_4': {'Pay_duly': 0, 'Pay_delay>=1': 1},
        'PAY_5': {'Pay_duly': 0, 'Pay_delay>=1': 1},
        'PAY_6': {'Pay_duly': 0, 'Pay_delay>=1': 1}
    }

    # Replace categorical values with numeric values
    for col, mapping in categorical_data_map.items():
        replace_categorical_values(df, col, mapping)

    # Save the processed data to a new CSV file
    df.to_csv(output_csv_path, index=False)

    print("Processed data saved to", output_csv_path)

# Set the path to your original CSV file and the output CSV file
input_csv_path = '.\datasets\default_credit.csv'
output_csv_path = '.\datasets\processed_default_credit.csv'

# Preprocess the default_credit dataset
preprocess_default_credit(input_csv_path, output_csv_path)