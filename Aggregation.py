import pandas as pd
from pathlib import Path
import os

def aggregate(inputpath,outputhpath):
    
    # List all CSV files in the directory
    if os.path.exists(inputpath):
        csv_files = [file for file in os.listdir(inputpath) if file.endswith('.csv')]

    # Read each CSV file and store its contents in a list of DataFrames
        if(csv_files):
            dfs = []
            for file in csv_files:
                file_path = os.path.join(inputpath, file)  # Construct full file path
                df = pd.read_csv(file_path)
                dfs.append(df)

            # Concatenate all DataFrames into one
            merged_df = pd.concat(dfs)
            # for df_to_merge in dfs[1:]:
            #     merged_df = merged_df.merge(df_to_merge, left_index=True, right_index=True, on=['a', 'b', 'c'])

            # Group by the row names and calculate the average of each field
            aggregated_df = merged_df.groupby(['method_name','espilon','k']).mean()
            # aggregated_df.reset_index(inplace=True)
            # Write the aggregated data to a new CSV file
            aggregated_df.to_csv(outputhpath)
        else:
            print(f"No CSV files found in {str(inputpath)}")
  


dataset = 'default_credit' #'compas','adult'
# for dataset in ('adult'): #,'hospital','informs'
for model in ('RF', 'NN','SVM'): # 'XGBoost' ,
    inputpath = Path('./dpnice/optimized/visualized/{}/{}/'.format(dataset,model))

    outputpath = './dpnice/optimized/visualized/{}/{}/aggregated_stats.csv'.format(dataset,model)
    aggregate(inputpath,outputpath)

        