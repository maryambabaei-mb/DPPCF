import pandas as pd
import argparse
import dill
import os
# import ast
import sys
import numpy as np
# import matplotlib.pyplot as plt
# from utils.analysis.draw_plots import draw_CDF
from tools.save_results import save_CF_to_csv
from utils.analysis.draw_plots import plot_utility_privacy_tradeoff,plot_utility_privacy_tradeoff_3_in_1
EXP_COUNT = 2000

def generate_statistics_record_new(array_name,epsilon,k,statistics_array):
    average_distance = np.mean(statistics_array[:, 0])
    average_reidentification_rate = np.mean(statistics_array[:, 1])
    not_matching_count = ((len(statistics_array[:, 1]) -  np.count_nonzero(statistics_array[:, 1]))/len(statistics_array[:, 1]))*100
    one_exact_exact_match = (np.count_nonzero(statistics_array[:, 1]==1)/len(statistics_array[:, 1]))*100
    average_changed_rate = np.mean(statistics_array[:, 2])
    average_same_inst_rate = np.mean(statistics_array[:, 3])
    average_kl_divergence = np.mean(statistics_array[:, 4])
    ####k = statistics_array[:, 5]
    average_CF_min_dist = np.mean(statistics_array[:, 6])    
    averagE_CF_min_k_dist = np.mean(statistics_array[:, 7])
    average_CF_rand_k_dist = np.mean(statistics_array[:, 8])
    
    std_distance = np.std(statistics_array[:, 0])
    std_reidentification_rate = np.std(statistics_array[:, 1])
    std_changed_rate = np.std(statistics_array[:, 2])
    std_same_inst_rate = np.std(statistics_array[:, 3])
    success_rate = ( len(statistics_array) / EXP_COUNT ) * 100
    Have_match_count = 100 - not_matching_count #(calculates everyting as percentage)-

    
    std_CF_min_k_dist = np.std(statistics_array[:, 7])   
    record = {'method_name': array_name,'espilon': epsilon ,'k' : k,'average_distance': average_distance,
              'average_CF_min_dist':average_CF_min_dist,'average_CF_min_k_dist':averagE_CF_min_k_dist,'average_CF_rand_k_dist':average_CF_rand_k_dist,
              'average_kl_divergence':average_kl_divergence, 'average_reidentification_rate': average_reidentification_rate, 'average_changed_rate': average_changed_rate,
              'average_same_inst_rate': average_same_inst_rate, 'std_distance': std_distance, 'std_reidentification_rate': std_reidentification_rate,
              'std_changed_rate': std_changed_rate,'std_same_inst_rate':std_same_inst_rate,'not_matching_count':not_matching_count,'one_exact_exact_match':one_exact_exact_match ,
              'Have_match_count':Have_match_count,  'success_rate': success_rate,'std_CF_min_k_dist':std_CF_min_k_dist}
    return record

def generate_statistics_record_new_correct(array_name,epsilon,k,statistics_array):
    average_distance = np.mean(statistics_array[:, 0])
    average_reidentification_rate = np.mean(statistics_array[:, 1])
    not_matching_count = len(statistics_array[:, 1]) -  np.count_nonzero(statistics_array[:, 1])
    one_exact_exact_match = np.count_nonzero(statistics_array[:, 1]==1)
    average_changed_rate = np.mean(statistics_array[:, 2])
    average_same_inst_rate = np.mean(statistics_array[:, 3])
    average_kl_divergence = np.mean(statistics_array[:, 4])
    ####k = statistics_array[:, 5]
    average_CF_min_dist = np.mean(statistics_array[:, 6])    
    averagE_CF_min_k_dist = np.mean(statistics_array[:, 7])
    average_CF_rand_k_dist = np.mean(statistics_array[:, 8])
    
    std_distance = np.std(statistics_array[:, 0])
    std_reidentification_rate = np.std(statistics_array[:, 1])
    std_changed_rate = np.std(statistics_array[:, 2])
    std_same_inst_rate = np.std(statistics_array[:, 3])
    success_rate = len(statistics_array) / 100
    Have_match_count = (success_rate * 100) - not_matching_count
    record = {'method_name': array_name,'espilon': epsilon ,'k' : k,'average_distance': average_distance,
              'average_CF_min_dist':average_CF_min_dist,'average_CF_min_k_dist':averagE_CF_min_k_dist,'average_CF_rand_k_dist':average_CF_rand_k_dist,
              'average_kl_divergence':average_kl_divergence, 'average_reidentification_rate': average_reidentification_rate, 'average_changed_rate': average_changed_rate,
              'average_same_inst_rate': average_same_inst_rate, 'std_distance': std_distance, 'std_reidentification_rate': std_reidentification_rate,
              'std_changed_rate': std_changed_rate,'std_same_inst_rate':std_same_inst_rate,'not_matching_count':not_matching_count,'one_exact_exact_match':one_exact_exact_match ,
              'Have_match_count':Have_match_count,  'success_rate': success_rate}
    return record



def util_piv_metrics(record):

    # Extracting the specified values from the record dictionary
    util_piv_array = [
        # epsilon,
        # k,
        record['average_distance'],
        record['average_reidentification_rate'],
        record['not_matching_count'],
        record['one_exact_exact_match'],
        record['average_CF_min_dist'],
        record['average_CF_min_k_dist'],
        record['average_CF_rand_k_dist'],
        record['success_rate'],
        record['Have_match_count']
    ]
    return util_piv_array

def generate_method_df(df,CF_method):
    
    for data_dict in df:
        outputdf = {}
        
        # iterate over df, get all arrays related to CF_method and assign it into the output and generated df  
        for name, statistics_array in df.items():
            onedatarow = {} 
            if CF_method in name:
                array = np.array(statistics_array)
                onedatarow[name] = array
                outputdf.append(onedatarow) 
        return outputdf
    

if __name__ == '__main__':

    #### load parameters
    parser = argparse.ArgumentParser(description='Script pretraining bbox models')
    parser.add_argument('--dataset', type=str, default='informs', help='hospital,adult,informs,synth_adult')
    parser.add_argument('--rseed', type=int, default=2, help='random seed: choose between 0 - 5')
    parser.add_argument('--model', type=str, default='RF', help='NN, RF, SVM, XgBoost')

    # get input      
    args = parser.parse_args()
    dataset_name = args.dataset
    RANDOM_SEED = args.rseed
    model = args.model

    datasets = ['default_credit'] #,'default_credit','hospital','informs','hospital','compas','adult'
    models = ['NN','RF'] #,'XGBoost','SVM'
    CF_method_list = ['synth_dp_cf','NICE','LDP_CF','LDP_SRR','LDP_Noisy_max','inline_LDP','zerocost_DP_CF','ldp_ds_cf']

    for dataset_name in datasets:
        for model in models:
            for RANDOM_SEED in range(0,5): 

                epsilons = [0.01, 0.1, 1, 5, 10]
                ks = [3,5,10,20]    ##### open analysed file into an structure 
                CF_method_list = ['inline_LDP','zerocost_DP_CF','NICE','LDP_CF','LDP_SRR','LDP_Noisy_max','synth_dp_cf','ldp_ds_cf']
                n_count_list = [0,3,5,10,20]
                one_neighbour_methods = ['LDP_CF','LDP_SRR','LDP_Noisy_max','NICE','synth_dp_cf','ldp_ds_cf']
                multiple_neighbour_methods = ['inline_LDP','zerocost_DP_CF']

                ##### analyse all arrays, generate tables and graphs
                #### Save everything in verbos mode
                processed_dir = './dpnice/optimized/analysis/{}/'.format(dataset_name)
                processed_file = '{}{}_{}.pkl'.format(processed_dir,model,RANDOM_SEED)
                graphs_dir = './dpnice/optimized/visualized/{}/{}/'.format(dataset_name,model)
                stat_file_path = '{}/SEED_{}.csv'.format(graphs_dir,RANDOM_SEED)
                
                if not os.path.exists(graphs_dir):
                    os.makedirs(graphs_dir, exist_ok=True)
                
                

                if(os.path.exists(processed_file)):
                    with open(processed_file, 'rb') as file:
                            df = dill.load(file)
                    print(f"loaded dataset from{processed_file}")
                    
                        # first, extract NICE
                    # Now, Generate LDP_df
                    nice_df = []
                    for data_dict in df:            
                        # Access the name and components of each dictionary
                        onerow = {}
                        NICE_array_name = 'NICE_1_0' #.format(epsilon,k)
                        for name, statistics_array in data_dict.items():
                            if name == NICE_array_name:
                                onerow[name] = statistics_array
                                in_array = np.array(statistics_array)
                                #calculate statistics and add to csv_file
                                stat_analysis = generate_statistics_record_new("NICE",0,0,in_array)
                                save_CF_to_csv(stat_analysis, csv_file_path=stat_file_path)
                                nice_df.append(onerow)
                                nice_dp_utility_df = util_piv_metrics(stat_analysis)

                    LDP_df = []
                    laplace_noise_dp_utility_df = [[] for _ in range(len(epsilons))]
                    for data_dict in df:            
                        # Access the name and components of each dictionary
                        for eps_ind in range(len(epsilons)):
                            onerow = {}
                            ldp_array_name = 'LDP_CF_{}_0'.format(epsilons[eps_ind])
                            for name, statistics_array in data_dict.items():
                                if name == ldp_array_name:
                                    onerow[name] = statistics_array
                                    in_array = np.array(statistics_array)
                                    #calculate statistics and add to csv_file
                                    stat_analysis = generate_statistics_record_new("laplace_noise_DP",epsilons[eps_ind],'0',in_array)
                                    save_CF_to_csv(stat_analysis, csv_file_path=stat_file_path)
                                    LDP_df.append(onerow)
                                    laplace_noise_dp_utility_df[eps_ind]= util_piv_metrics(stat_analysis)
                    # plot_utility_privacy_tradeoff_3_in_1(laplace_noise_dp_utility_df,graphs_dir,RANDOM_SEED,epsilons,nice_param=nice_dp_utility_df,method_name='laplace_noise_DP')

                    # Generate SRR_DP_CF
                    SRR_df = []
                    EXP_dp_dp_utility_df = [[] for _ in range(len(epsilons))]
                    for data_dict in df:            
                        # Access the name and components of each dictionary
                        for eps_ind in range(len(epsilons)):
                            
                            onerow = {}
                            SRR_array_name = 'LDP_SRR_{}_0'.format(epsilons[eps_ind])
                            for name, statistics_array in data_dict.items():
                                if name == SRR_array_name:
                                    onerow[name] = statistics_array
                                    in_array = np.array(statistics_array)
                                    #calculate statistics and add to csv_file
                                    stat_analysis = generate_statistics_record_new("DP_EXP_Feature",epsilons[eps_ind],'0',in_array)
                                    save_CF_to_csv(stat_analysis, csv_file_path=stat_file_path)
                                    SRR_df.append(onerow)
                                    EXP_dp_dp_utility_df[eps_ind]= util_piv_metrics(stat_analysis)
                    # plot_utility_privacy_tradeoff_3_in_1(EXP_dp_dp_utility_df,graphs_dir,RANDOM_SEED,epsilons,nice_param=nice_dp_utility_df,method_name='DP_EXP_Feature')
                    


                    
                    Noisy_max_df = []
                    Noisy_max_dp_ds_dp_utility_df = [[] for _ in range(len(epsilons))]
                    for data_dict in df:            
                        # Access the name and components of each dictionary
                        for eps_ind in range(len(epsilons)):
                            
                            onerow = {}
                            Noisy_max_array_name = 'LDP_Noisy_max_{}_0'.format(epsilons[eps_ind])
                            for name, statistics_array in data_dict.items():
                                if name == Noisy_max_array_name:
                                    onerow[name] = statistics_array
                                    in_array = np.array(statistics_array)
                                    #calculate statistics and add to csv_file
                                    stat_analysis = generate_statistics_record_new("Noisy_Max_DP",epsilons[eps_ind],'0',in_array)
                                    save_CF_to_csv(stat_analysis, csv_file_path=stat_file_path)
                                    Noisy_max_df.append(onerow)
                                    Noisy_max_dp_ds_dp_utility_df[eps_ind]= util_piv_metrics(stat_analysis)
                    # plot_utility_privacy_tradeoff_3_in_1(Noisy_max_dp_ds_dp_utility_df,graphs_dir,RANDOM_SEED,epsilons,nice_param=nice_dp_utility_df,method_name='Noisy_Max_DP')

                    # 'synth_dp_cf'
                    synth_dp_df = []
                    synth_dp_ds_dp_utility_df = [[] for _ in range(len(epsilons))]
                    for data_dict in df:            
                        # Access the name and components of each dictionary
                        for eps_ind in range(len(epsilons)):
                            
                            onerow = {}
                            synth_dp_array_name = 'synth_dp_cf_{}_0'.format(epsilons[eps_ind])
                            for name, statistics_array in data_dict.items():
                                if name == synth_dp_array_name:
                                    onerow[name] = statistics_array
                                    in_array = np.array(statistics_array)
                                    #calculate statistics and add to csv_file
                                    stat_analysis = generate_statistics_record_new("synth_dp_ds",epsilons[eps_ind],'0',in_array)
                                    save_CF_to_csv(stat_analysis, csv_file_path=stat_file_path)
                                    synth_dp_df.append(onerow)
                                    synth_dp_ds_dp_utility_df[eps_ind]= util_piv_metrics(stat_analysis)
                    # plot_utility_privacy_tradeoff_3_in_1(synth_dp_ds_dp_utility_df,graphs_dir,RANDOM_SEED,epsilons,nice_param=nice_dp_utility_df,method_name='Synth_DP_DS')

                    # 'ldp_ds_cf'
                    ldp_ds_df = []
                    dp_ds_dp_utility_df = [[] for _ in range(len(epsilons))]
                    for data_dict in df:            
                        # Access the name and components of each dictionary
                        for eps_ind in range(len(epsilons)):
                            
                            onerow = {}
                            synth_dp_array_name = 'ldp_ds_cf_{}_0'.format(epsilons[eps_ind])
                            for name, statistics_array in data_dict.items():
                                if name == synth_dp_array_name:
                                    onerow[name] = statistics_array
                                    in_array = np.array(statistics_array)
                                    #calculate statistics and add to csv_file
                                    stat_analysis = generate_statistics_record_new("DP_DS",epsilons[eps_ind],'0',in_array)
                                    save_CF_to_csv(stat_analysis, csv_file_path=stat_file_path)
                                    ldp_ds_df.append(onerow)
                                    dp_ds_dp_utility_df[eps_ind]= util_piv_metrics(stat_analysis)
                    # plot_utility_privacy_tradeoff_3_in_1(dp_ds_dp_utility_df,graphs_dir,RANDOM_SEED,epsilons,nice_param= nice_dp_utility_df,method_name='DP_DS')

                    inline_LDP_df = []
                    inlinedp_utility_df = [[[] for _ in range(len(ks))] for _ in range(len(epsilons))] #np.empty((len(epsilons), len(ks)))
                    for data_dict in df:            
                        # Access the name and components of each dictionary
                        for eps_ind in range(len(epsilons)):
                            for k_ind in range(len(ks)):
                                onerow = {}
                                LDP_inline_array_name = 'inline_LDP_{}_{}'.format(epsilons[eps_ind],ks[k_ind])
                                for name, statistics_array in data_dict.items():
                                    if name == LDP_inline_array_name:
                                        onerow[name] = statistics_array
                                        in_array = np.array(statistics_array)
                                        #calculate statistics and add to csv_file
                                        stat_analysis = generate_statistics_record_new("inline_DP",epsilons[eps_ind],ks[k_ind],in_array)
                                        save_CF_to_csv(stat_analysis, csv_file_path=stat_file_path)
                                        inline_LDP_df.append(onerow)
                                        inlinedp_utility_df[eps_ind][k_ind]= util_piv_metrics(stat_analysis)
                    
                    # plot_utility_privacy_tradeoff_3_in_1(inlinedp_utility_df,graphs_dir,RANDOM_SEED,epsilons,ks,nice_param=nice_dp_utility_df,method_name='Inline_DP')

                    # Generate Zero_cost_DP
                    zerocost_DP_df = []
                    zerocost_utility_df = [[[] for _ in range(len(ks))] for _ in range(len(epsilons))] #np.empty((len(epsilons), len(ks)))
                    for data_dict in df:            
                        # Access the name and components of each dictionary
                        for eps_ind in range(len(epsilons)):
                            for k_ind in range(len(ks)):
                                onerow = {}
                                
                                zerocost_DP_CF_array_name = 'zerocost_DP_CF_{}_{}'.format(epsilons[eps_ind],ks[k_ind])
                                for name, statistics_array in data_dict.items():
                                    if name == zerocost_DP_CF_array_name:
                                        onerow[name] = statistics_array
                                        in_array = np.array(statistics_array)
                                        #calculate statistics and add to csv_file
                                        stat_analysis = generate_statistics_record_new("Exp_prox_DP",epsilons[eps_ind],ks[k_ind],in_array)
                                        save_CF_to_csv(stat_analysis, csv_file_path=stat_file_path)
                                        zerocost_DP_df.append(onerow)
                                        # UP_array[name] = util_piv_metrics(epsilon,k,stat_analysis)
                                        zerocost_utility_df[eps_ind][k_ind]= util_piv_metrics(stat_analysis)
                    # Generate Zero_cost_DP
                    #### Now all statistics are saved, all dfs are generated, just draw graphs and save each group
                    
                    # plot_utility_privacy_tradeoff_3_in_1(zerocost_utility_df,graphs_dir,RANDOM_SEED,epsilons,ks,nice_param=nice_dp_utility_df,method_name='Exp_prox_DP')
                    print("Dfs generated")


