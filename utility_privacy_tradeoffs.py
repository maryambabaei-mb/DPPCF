import pandas as pd
import argparse
import dill
import os
import ast
import sys


if __name__ == '__main__':

    #### load parameters
    parser = argparse.ArgumentParser(description='Script pretraining bbox models')
    parser.add_argument('--dataset', type=str, default='adult', help='hospital,adult,informs,synth_adult')
    parser.add_argument('--rseed', type=int, default=2, help='random seed: choose between 0 - 5')
    parser.add_argument('--model', type=str, default='RF', help='NN, RF, SVM, XgBoost')

    # get input      
    args = parser.parse_args()
    dataset_name = args.dataset
    RANDOM_SEED = args.rseed
    model = args.model


    
    # #### test data to check if it works
    # dataset_name = 'adult' #'synth_adult' #'hospital'#'adult'
    # RANDOM_SEED = 2
    # model = 'RF'
    # epsilon = 1
    # NEIGHBOR_COUNT = 0
    # # instance_num = 12
    # CF_method = 'NICE'#'zerocost_DP_CF' #'DP_CF'#'LDP_CF'


    stats_dir =    './dpnice/U_P_tradeoff/{}/{}/{}/'.format(dataset_name,model,RANDOM_SEED)
    stat_graph = '{}{}_{}.pkl'.format(stats_dir,model,RANDOM_SEED)  # privacy/utility tradeoff
    # processed_dir = './dpnice/srr_processed_results/{}/'.format(dataset_name)
    # processed_file = '{}{}_{}.pkl'.format(processed_dir,model,RANDOM_SEED)
    graphs_dir = './dpnice/srr_graphs_dir/{}/{}/'.format(dataset_name,model)    # path where we saved the final results
    stat_file_path = '{}/SEED_{}.csv'.format(graphs_dir,RANDOM_SEED)    # final results file generated 
    
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir, exist_ok=True)

    
#    for RANDOM_SEED in range (0 ,6):    #open resutls of every seed, process them and save them in your file
    all_datasets = []
    if(os.path.exists(stat_file_path)):
        df = pd.read_csv(stat_file_path)
        df = df.dropna(how='all')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        for index,row in df.iterrows():
            for epsilon in (0.01,0.1,1,5,10):
                # resultfilepath = '{}{}_{}_eps_{}.csv'.format(resultsdir, model, RANDOM_SEED,epsilon)
                for CF_method in ('LDP_CF','LDP_SRR','LDP_Noisy_max','inline_LDP','NICE','zerocost_DP_CF','synth_dp_cf','ldp_ds_cf'):
                    for NEIGHBOR_COUNT in (0,3,5,10,20):
                    
                        #First, generate array names, to know in file you should search for what
                        #generate array name
                        ## here we have a problem for k, so we should add 0 to k and epsilon to be able to process files correctly
                        if CF_method in ('LDP_CF','LDP_SRR','LDP_Noisy_max','inline_LDP','NICE','zerocost_DP_CF','synth_dp_cf','ldp_ds_cf' ):
                        #search in file for all related data
                            array_name = '{}_{}_{}'.format(CF_method,epsilon,NEIGHBOR_COUNT)
                            #add its name to the structure 
                            # iterate over all df, read data, and extract information related to the array into that.
                            statistics_array =[]
                            
                            for index, row in df.iterrows():
                                if (CF_method in ('LDP_CF','LDP_SRR','LDP_Noisy_max','NICE','synth_dp_cf','ldp_ds_cf') and NEIGHBOR_COUNT == 0) or (CF_method in ('inline_LDP','zerocost_DP_CF') and NEIGHBOR_COUNT != 0):
                                    if NEIGHBOR_COUNT == 0 : # one dimensional array
                                        # if nice: only one setting
                                        if (CF_method == 'NICE'):
                                            
                                            
                                            # if pre-procc and post-pross ldp: 5 settings (5 epsilon values)
                                        
                                    else : #2 dimensional array 
                                        {
                                            # inline_ ldp and exp_mech: 5 * 4 settings (5 epsilon values and 4 k values)
                                        }
                                    if row['CF_method'] == CF_method and row['epsilon'] == epsilon  and row['k'] == NEIGHBOR_COUNT:
                                        #if the instance is a counterfactual, then add its data to the array
                                        if row['not_cf'] == False:
                                            # Access values from other columns for the rows with the special value
                                            distance = row['CF_distance']
                                            CF_distance = ast.literal_eval(distance)[0]
                                            reidentification_rate = row['cf__ri_count']
                                            changed_rate = row['changed_rate']
                                            same_inst_rate = row['same_inst_rate']
                                            kl_divergence = row['KL_divergence']
                                            min_dist = row['CF_min_dist']
                                            CF_min_dist = ast.literal_eval(min_dist)[0]
                                            min_k_dist = row['CF_min_k_dist']
                                            CF_min_k_dist = ast.literal_eval(min_k_dist)[0]
                                            rand_k_dist = row['CF_rand_k_dist']
                                            CF_rand_k_dist = ast.literal_eval(rand_k_dist)[0]
                                            k = row['k']
                                            statistics_array.append([CF_distance,reidentification_rate,changed_rate,same_inst_rate,kl_divergence,k,CF_min_dist,CF_min_k_dist,CF_rand_k_dist])
                                            
                                        ### to make sure about the sizez, we can add zero/NA arrays for non_CF instances
                                    # else:
                            #### search in result file, if this array exsists, just append the data part, otherwise insert array    
                            if(statistics_array != []):
                                onedatarow = {}
                                onedatarow[array_name] = statistics_array
                                all_datasets.append(onedatarow)
                                print(f"file {resultfilepath} Processed")
                else:
                    print(f"file {resultfilepath} does not exist")
                    continue