# import pandas as pd
import requests
from pmlb import fetch_data
# from io import BytesIO
import os

# Define dataset names
classification_dataset_names = ['adult']  # Replace [...] with your actual classification dataset names
#regression_dataset_names = [...]  # Replace [...] with your actual regression dataset names
dataset_names =  ['adult'] # classification_dataset_names + regression_dataset_names

# Define GitHub URL and suffix
GITHUB_URL = 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets'
suffix = '.tsv.gz'

# Function to download and save dataset
def download_and_save_dataset(dataset_name):
    url = f'{GITHUB_URL}/{dataset_name}{suffix}'
    response = requests.get(url)

    if response.status_code == 200:
        # Load the data into a pandas DataFrame
        df = fetch_data(name)
        #df = pd.read_csv(BytesIO(response.content), compression='gzip', sep='\t')

        # Save DataFrame to a local CSV file
        csv_filename = f'{dataset_name}.csv'
        df.to_csv(csv_filename, index=False)

        print(f'{dataset_name} downloaded and saved as {csv_filename}')
    else:
        print(f'Failed to download {dataset_name}')

# Create a directory to store the downloaded CSV files
if not os.path.exists('datasets'):
    os.makedirs('datasets')

# Download and save each dataset
for dataset_name in dataset_names:
    #download_and_save_dataset(dataset_name)
    df = fetch_data(dataset_name)
    csv_filename = f'{dataset_name}.csv'
    df.to_csv(csv_filename, index=False)

    print(f'{dataset_name} downloaded and saved as {csv_filename}')
    # else:
    #     print(f'Failed to download {dataset_name}')
