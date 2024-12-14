import pandas as pd
import random

def replace_categorical_values(dataset,column, mapping):
    dataset[column] = dataset[column].replace(mapping)
    
def preprocess(input_csv_path):
    orig_data = pd.read_csv(input_csv_path)
    orig_data.dropna()
    orig_data.reset_index(drop=True, inplace=True)
    orig_data['Length of Stay']=pd.to_numeric(orig_data['Length of Stay'], errors ='coerce').fillna(120).astype('int')
    orig_data['Total Costs'] = orig_data['Total Costs'].replace('[\$,]', '', regex=True).astype(float)
    orig_data['Total Charges'] = orig_data['Total Charges'].replace('[\$,]', '', regex=True).astype(float)
    orig_data = orig_data.drop(columns=['Operating Certificate Number', 'Facility Id', 'Facility Name', 'CCS Diagnosis Description', 'CCS Procedure Description',  'APR DRG Description','APR MDC Description','APR Severity of Illness Description'])
    feat_names =  list(orig_data.columns)
    print(feat_names)

 

    #numerical_columns = ['Numerical_Column_1', 'Numerical_Column_2']
    #'CCS Diagnosis Code','CCS Procedure Code', 'APR DRG Code','APR MDC Code' ,'APR Severity of Illness Code',
    categorical_columns = ['Health Service Area','Hospital County', 'Age Group','Zip Code - 3 digits','Gender','Race','Ethnicity','Length of Stay','Type of Admission', 'Patient Disposition','Discharge Year', 'APR Risk of Mortality','APR Medical Surgical Description','Payment Typology 1', 'Payment Typology 2','Payment Typology 3','Abortion Edit Indicator', 'Emergency Department Indicator']
    #    'Categorical_Column_1', 'Categorical_Column_2', 'Categorical_Column_3']

    categorical_data_map = {}
    for col in categorical_columns:
        unique_values = orig_data[col].unique()
        categorical_data_map[col] = {'values': unique_values}

    processed_data = orig_data.copy()
    for col, values_info in categorical_data_map.items():
        replace_categorical_values(processed_data,col, dict(zip(values_info['values'], range(1, len(values_info['values']) + 1))))

    processed_data.to_csv('processed_hospital.csv')

# Set the path to your original CSV file
input_csv_path = 'dpnice\Hospital_Inpatient_Discharges__SPARCS_De-Identified___2015_20240220.csv'

# Set the path for the new CSV file with the randomly chosen instances
output_csv_path = 'hospital_discharge.csv'

# Set the number of instances you want to choose randomly
num_instances_to_choose = 80000

# Read the original CSV file into a pandas DataFrame
original_data = pd.read_csv(input_csv_path)

# Get the total number of records in the dataset
total_records = len(original_data)

# Check if the dataset is smaller than the number of instances you want to choose
if num_instances_to_choose >= total_records:
    print("Error: The dataset is smaller than the number of instances you want to choose.")
    exit()

# Choose random indices for the instances you want to select
random_indices = random.sample(range(total_records), num_instances_to_choose)

# Create a new DataFrame with the randomly chosen instances
random_data = original_data.iloc[random_indices]

# Save the new DataFrame to a new CSV file
random_data.to_csv(output_csv_path, index=False)

print(f"Randomly chosen {num_instances_to_choose} instances saved to {output_csv_path}")

preprocess(output_csv_path)






