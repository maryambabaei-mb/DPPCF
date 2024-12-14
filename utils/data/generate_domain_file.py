import pandas as pd
import json
from load_data import Fetcher

def get_feature_types(df):
            feature_names = list(df.columns)
            X_na= df.dropna()
            con_feat = []
            cat_feat =[]
            # idx = 0
            for feature_name in feature_names:
                x = X_na[feature_name].copy()
                if  not all(float(i).is_integer() for i in x.unique()):
                    #con_feat.append(feature_name)
                    con_feat.append(feature_name)
                elif x.nunique() > 10:
                    #con_feat.append(feature_name)
                    con_feat.append(feature_name)
                else:
                    #cat_feat.append(feature_name)
                    cat_feat.append(feature_name)
                # idx +=1
            return con_feat, cat_feat

def generate_domain_file_1(csv_file, json_file):
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        print("CSV file loaded successfully.")
        
        # Initialize a dictionary to store the domain info
        domain_dict = {}
        
        # Iterate over the columns of the DataFrame
        for column in df.columns:
            # Determine if the column is categorical or numerical
            if pd.api.types.is_numeric_dtype(df[column]):
                # If numerical, store the range as domain
                domain_dict[column] = {
                    'type': 'numerical',
                    'min': int(df[column].min()),  # Convert to native Python int
                    'max': int(df[column].max())   # Convert to native Python int
                }
                print(f"Processed numerical column: {column}")
            else:
                # If categorical, store the unique values as domain
                domain_dict[column] = {
                    'type': 'categorical',
                    'values': df[column].astype(str).unique().tolist()
                }
                print(f"Processed categorical column: {column}")
        
        # Write the domain dictionary to a JSON file
        with open(json_file, 'w') as json_f:
            json.dump(domain_dict, json_f, indent=4)
        print(f"Domain file '{json_file}' created successfully.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

def generate_domain_file(csv_file, json_file):
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        print("CSV file loaded successfully.")
        
        # Initialize a dictionary to store the domain info
        domain_dict = {}
        
        # Iterate over the columns of the DataFrame
        for column in df.columns:
            unique_values = df[column].unique()
            unique_count = len(unique_values)
            
            # Determine if the column should be 'finite'
            if unique_count < 100:
                # Classify as 'finite' with all possible values
                domain_dict[column] = {
                    "name": column,
                    "type": "finite",
                    "representation": [str(val) for val in unique_values]
                }
                print(f"Processed finite column: {column}")
            else:
                # Determine if the column is categorical or numerical
                if pd.api.types.is_numeric_dtype(df[column]):
                    # If numerical, store the range as domain
                    domain_dict[column] = {
                        'name': column,
                        'type': 'integer',
                        'representation': 'integer',
                    }
                    print(f"Processed numerical column: {column}")
                else:
                    # If categorical, store the unique values as domain
                    domain_dict[column] = {
                        'name': column,
                        'type': 'categorical',
                        'values': df[column].astype(str).unique().tolist()
                    }
                    print(f"Processed categorical column: {column}")
        
        # Wrap the dictionary in the "columns" key
        final_output = {
            "columns": list(domain_dict.values())
        }
        
        # Write the final output to a JSON file
        with open(json_file, 'w') as json_f:
            json.dump(final_output, json_f, indent=4)
        print(f"Domain file '{json_file}' created successfully.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

        # Write the domain dictionary to a JSON file
    #     with open(json_file, 'w') as json_f:
    #         json.dump(domain_dict, json_f, indent=4)
    #     print(f"Domain file '{json_file}' created successfully.")
    
    # except Exception as e:
    #     print(f"An error occurred: {e}")

def generate_domain_file_for_mst(csv_file, json_file):
    try:
        df = pd.read_csv(csv_file)
        
        domain_list = []
        
        for column in df.columns:
            unique_values = df[column].unique()
            unique_count = len(unique_values)
            
            if pd.api.types.is_numeric_dtype(df[column]):
                column_info = {
                    "name": column,
                    "type": "integer",
                    "representation": "integer"
                }
            elif unique_count < 100:
                column_info = {
                    "name": column,
                    "type": "finite",
                    "representation": [str(val) for val in unique_values]
                }
            else:
                column_info = {
                    "name": column,
                    "type": "categorical",
                    "values": [str(val) for val in unique_values]
                }
            
            domain_list.append(column_info)
        
        final_output = {
            "columns": domain_list
        }
        
        with open(json_file, 'w') as json_f:
            json.dump(final_output, json_f, indent=4)
    
    except Exception as e:
        print(f"An error occurred: {e}")



def convert_csv_separators(input_file, output_file):
    # Load the CSV file with semicolon separator
    df = pd.read_csv(input_file, sep=';')
    
    # Save the DataFrame to a new CSV file with comma separator
    df.to_csv(output_file, sep=',', index=False)


def generate_domain_from_csv(csv_path, json_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize the domain dictionary
    domain = {'columns': []}
    
    # Analyze each column
    for column in df.columns:
        col_data = df[column]
        col_type = col_data.dtype
        
        # Determine the data type
        if pd.api.types.is_integer_dtype(col_type):
            data_type = 'integer'
        elif pd.api.types.is_float_dtype(col_type):
            data_type = 'float'
        elif pd.api.types.is_string_dtype(col_type):
            data_type = 'string'
        else:
            data_type = 'unknown'
        
        # Add column metadata to the domain dictionary
        domain['columns'].append({
            'name': column,
            'type': data_type
        })
    
    # Write the domain dictionary to a JSON file
    with open(json_path, 'w') as json_file:
        json.dump(domain, json_file, indent=4)


# Define the path to the CSV file and the output JSON file
dataset_name = 'compas'
csv_file = './dpnice/datasets/{}.csv'.format(dataset_name) # hospital.csv'
# new_csv_file = './dpnice/datasets/{}_new.csv'.format(dataset_name)
json_file = './dpnice/jsons/{}_domain.json'.format(dataset_name) 
# convert_csv_separators(csv_file,new_csv_file)

generate_domain_file(csv_file, json_file)
# generate_domain_from_csv(new_csv_file, json_file)