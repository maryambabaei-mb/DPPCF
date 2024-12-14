import csv
import os
import numpy as np

def save_results_to_csv(results, csv_file_path='results.csv'):
    # Check if the file already exists
    directory = os.path.dirname(csv_file_path)

    # Check if the directory exists, create it if not
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    file_exists = os.path.exists(csv_file_path)

    # Open the CSV file in append mode
    with open(csv_file_path, 'a+') as csvfile:

        # Define the CSV writer
        if csvfile.tell() == 0:  # Check if the file is empty
            csv_writer = csv.DictWriter(csvfile, fieldnames=['dataset', 'RANDOM_SEED', 'model', 'epsilon','accuracy'])
            csv_writer.writeheader()
        # Write the header only if the file is newly created
        
        # Write each record
        csv_writer = csv.DictWriter(csvfile, fieldnames=['dataset', 'RANDOM_SEED', 'model', 'epsilon','accuracy'])
        csv_writer.writerows(results)


# def generate_CF_record(CF_method,df,CF_distance,cf__ri_count,not_cf=False,epsilon = 0,k=0,changed_rate=0,not_changed_count=0):
#     # result_file = './dpnice/results/models_accuracy.csv'
    
#     df_csv_string = df.to_csv(index=False)
#     record = {'CF_method': CF_method, 'epsilon': epsilon, 'k':k ,'CF_distance': CF_distance,'cf__ri_count':cf__ri_count,'not_cf':not_cf,'changed_rate':changed_rate,
#               'not_changed_count':not_changed_count,'df': df_csv_string}
#     #save_results_to_csv(model_info, result_file)
#     return record


# def save_CF_to_csv(record, csv_file_path='CFs.csv'):
#     # Check if the file already exists
#     directory = os.path.dirname(csv_file_path)

#     # Check if the directory exists, create it if not
#     if not os.path.exists(directory):
#         os.makedirs(directory)
        
#     #extract fieldnames
#     if isinstance(record, dict):
#         fieldnames = record.keys()
    

#         #file_exists = os.path.exists(csv_file_path)

#         # Open the CSV file in append mode
#         with open(csv_file_path, 'a+') as csvfile:

#             # Define the CSV writer
#             if csvfile.tell() == 0:  # Check if the file is empty
#                 csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#                 csv_writer.writeheader()
#             # Write the header only if the file is newly created
            
#             # Write each record
#             csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             csv_writer.writerows(record)

#     else:
#         print("not CSV format Record")
        






def generate_CF_record(CF_method, df, CF_distance, CF_min_dist = 'NA' , CF_min_k_dist = 'NA' , CF_rand_k_dist = 'NA', cf__ri_count='NA', not_cf=False, epsilon=0, k=0, SameAsNice='NA', same_inst_rate='NA',changed_rate='NA',KL_divergence='NA',index = 0, iter=0):
    df_csv_string = df.to_string(index=False)
    record = {'CF_method': CF_method, 'epsilon': epsilon, 'k': k, 'CF_distance': CF_distance,
              'CF_min_dist' :CF_min_dist  , 'CF_min_k_dist' : CF_min_k_dist , 'CF_rand_k_dist' : CF_rand_k_dist , 
              'cf__ri_count': cf__ri_count, 'not_cf': not_cf, 'SameAsNice': SameAsNice,
              'changed_rate': changed_rate,'same_inst_rate':same_inst_rate, 'KL_divergence':KL_divergence,'index' : index, 'iter':iter,'df': df_csv_string}
    return record

def save_CF_to_csv(record, csv_file_path='CFs.csv'):
    # Check if the file already exists
    directory = os.path.dirname(csv_file_path)

    # Check if the directory exists, create it if not
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Open the CSV file in append mode
    with open(csv_file_path, 'a+') as csvfile:
        
        if isinstance(record, dict):
            fieldnames = record.keys()
            if csvfile.tell() == 0:  # Check if the file is empty
                #fieldnames = record.keys()
                csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                csv_writer.writeheader()
            # Use writerow for a single record
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csv_writer.writerow(record)
        else:
            print("Not a CSV format record")
            with open(csv_file_path, 'a+') as file:
                file.write(record+"\n")

        # Define the CSV writer
        

        # Write each record
        # csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        

def add_to_CF_file(file_name, CF, columns_list,ismember = False):
    directory = os.path.dirname(file_name)

    # Check if the directory exists, create it if not
    if not os.path.exists(directory):
        os.makedirs(directory)

    
    # CFN = np.array(CF)
    CFN = np.append(np.array(CF),ismember)
    # Open the CSV file in append mode
    with open(file_name, 'a+') as csvfile:
        if csvfile.tell() == 0:  # Check if the file is empty
            #fieldnames = record.keys()
            columns = np.append(columns_list,'is_member')
            # columns_list.append('is_member')
            csv_writer = csv.DictWriter(csvfile, fieldnames=columns)
            csv_writer.writeheader()
        # Use writerow for a single record
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(CFN)

