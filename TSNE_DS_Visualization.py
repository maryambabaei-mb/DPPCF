import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import dill
import copy
import os
from dpmechanisms.local_dp import make_DS_LDP


def draw_TSNE(original_set,updated_set,original_labels,updated_labels,original_title,updated_title,file_path):
    tsne = TSNE(n_components=2, random_state=42)

    # Fit and transform the data for the original dataset
    X_tsne_original = tsne.fit_transform(original_set)
    X_tsne_updated = tsne.fit_transform(updated_set)

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the original dataset
    plot_tsne(X_tsne_original, original_labels, original_title, axes[0])

    # Plot the modified dataset
    plot_tsne(X_tsne_updated, updated_labels, updated_title, axes[1])

    # Display the plots
    plt.show()
    plt.savefig(grph_file_path)
    plt.clf()
    ### save plot


def plot_tsne(X_tsne, y, title, ax):
    # Scatter plot
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6)
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    ax.set_title(title)
    ax.set_xlabel('TSNE Component 1')
    ax.set_ylabel('TSNE Component 2')

def plot_tsne_one(X_tsne, Y, title, ax):
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y, cmap='viridis')
    ax.set_title(title)
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)

def read_visualize_cfs(cf_file,basic_cf_file,outputfile_name,title):
    cf_list = pd.read_csv(cf_file, sep=',', header=[0], on_bad_lines='skip')
    basic_cf_list = pd.read_csv(basic_cf_file, sep=',', header=[0], on_bad_lines='skip')
    



    CF_files_X = np.vstack((cf_list, basic_cf_list))
    CF_files_Y = np.vstack((np.ones(len(cf_list)), np.zeros(len(basic_cf_list))))

    tsne = TSNE(n_components=2, random_state=42)

    # Fit and transform the data for the original dataset
    X_tsne_cfs = tsne.fit_transform(CF_files_X)
    # X_tsne_updated = tsne.fit_transform(ldp_DS.X_train)

    fig, axes = plt.subplots(figsize=(14, 6))
    # Plot the original dataset
    plot_tsne_one(X_tsne_cfs, CF_files_Y, title, axes)
    
    
    # plt.show()
    plt.savefig(outputfile_name)
    plt.clf()

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script pretraining bbox models')
    parser.add_argument('--dataset', type=str, default='compas', help='compas,hospital,adult,informs,synth_adult')
    parser.add_argument('--rseed', type=int, default=2, help='random seed: choose between 0 - 5')
    parser.add_argument('--model', type=str, default='RF', help='NN, RF, SVM, XgBoost')
    parser.add_argument('--epsilon', type=float, default='10', help='.01, .1, 1, 5, 10')
    parser.add_argument('--cptype', type=float, default='1', help='0:compare cf files, 1:compare whole dataset')
    parser.add_argument('--n_count', type=int, default=3, help='3 5 10 20')
    parser.add_argument('--CF_method', type=str, default='inline_LDP', help='NICE zerocost_DP_CF dp_ds_cf synth_dp_cf LDP_CF LDP_SRR LDP_Noisy_max inline_LDP')

    
    args = parser.parse_args()
    dataset_name = args.dataset
    RANDOM_SEED = args.rseed
    model = args.model
    epsilon = args.epsilon
    cptype = args.cptype
    n_count = args.n_count
    CF_method = args.CF_method
     
    if epsilon >= 1:
        epsilon = int(epsilon)

    if cptype == 1:
        ###### draw whole dataset
        DSoutdir = './dpnice/optimized/datasets_loaded/{}/'.format(dataset_name)
        synth_DSoutdir = './dpnice/optimized/datasets_loaded/synth_{}/'.format(dataset_name)
        modeloutdir = './dpnice/optimized/pretrained/{}/'.format(dataset_name)
        graphs_dir = './dpnice/optimized/graphs_dir/{}/{}/'.format(dataset_name,model)
        
        model_name = '{}{}_{}.pkl'.format(modeloutdir, model, RANDOM_SEED)
        DS_name = '{}{}_{}.pkl'.format(DSoutdir, model, RANDOM_SEED)
        synth_DS_name = '{}{}_{}_{}.pkl'.format(synth_DSoutdir, model, RANDOM_SEED,epsilon)

        ldp_grph_file_path = '{}/TSNE_eps_{}_ldp.png'.format(graphs_dir,epsilon) 
        synth_grph_file_path = '{}/TSNE_eps_{}_synth.png'.format(graphs_dir,epsilon) 

        if not os.path.exists(graphs_dir):
            os.makedirs(graphs_dir)

        with open(model_name, 'rb') as file:
            NICE_model = dill.load(file)
                #load the dataset
        with open(DS_name, 'rb') as file:
            DS = dill.load(file)
            #select original instance
        with open(synth_DS_name, 'rb') as file:
            Synth_DS = dill.load(file)

        X_train = DS.X_train.copy()
        #### fenerate and draw TSNE for original and DP (laplacian noise and RR) datasets
        ldp_DS = copy.deepcopy(DS)
        ldp_DS.X_train = make_DS_LDP(X_train,DS.dataset['feature_ranges'],epsilon,ldp_mech=0,cat_type =0)

        O_ldp_X = np.vstack((X_train, ldp_DS.X_train))
        O_ldp_Y = np.vstack((np.ones_like(DS.y_train), np.zeros_like(DS.y_train)))

        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(O_ldp_X)
        # X_tsne_ldp = tsne.fit_transform(O_ldp_X)
        plt.figure(figsize=(8, 6))

        # Plot the first distribution (assuming first half of data)
        # sns.scatterplot(x=X_tsne[:len(O_ldp_X)//2, 0], y=X_tsne[:len(O_ldp_X)//2, 1], color='blue', alpha=0.3, label='Original dataset')

        # Plot the second distribution (assuming second half of data)
        # sns.scatterplot(x=X_tsne[len(O_ldp_X)//2:, 0], y=X_tsne[len(O_ldp_X)//2:, 1], color='red', alpha=0.3, label='DP dataset')

        # Add density plot for each distribution
        sns.kdeplot(x=X_tsne[:len(O_ldp_X)//2, 0], y=X_tsne[:len(O_ldp_X)//2, 1], color='blue', fill=True, cmap='Blues', alpha=0.4, label='Original dataset',bw_adjust=2.0)
        sns.kdeplot(x=X_tsne[len(O_ldp_X)//2:, 0], y=X_tsne[len(O_ldp_X)//2:, 1], color='red', fill=True, cmap='Reds',  alpha=0.4,label='DP dataset',bw_adjust=2.0)
        # Add density plot for each distribution
        
        # sns.kdeplot(X_tsne[:len(O_ldp_X)//2, 0], X_tsne[:len(O_ldp_X)//2, 1], color='blue', shade=True, cmap='Blues', label=None)
        # sns.kdeplot(X_tsne[len(O_ldp_X)//2:, 0], X_tsne[len(O_ldp_X)//2:, 1], color='red', shade=True, cmap='Reds', label=None)

        # Adjust labels and legensynth_grph_file_pathd
        plt.title('DP-Dataset Vs. real dataset Distributions')
        # plt.xlabel('t-SNE Component 1')
        # plt.ylabel('t-SNE Component 2')
        plt.legend()
        ################# worked
        plt.savefig(ldp_grph_file_path)
        plt.clf()
        ################# worked
        # draw TSNE for original and synthetic datasets
        # ldp_DS = copy.deepcopy(DS)
        # ldp_DS.X_train = make_DS_LDP(X_train,DS.dataset['feature_ranges'],epsilon,ldp_mech=0,cat_type =0)

        synth_ldp_X = np.vstack((X_train, Synth_DS.X_train))
        synth_ldp_Y = np.vstack((np.ones_like(DS.y_train), np.zeros_like(DS.y_train)))

        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(synth_ldp_X)
        # X_tsne_ldp = tsne.fit_transform(O_ldp_X)
        plt.figure(figsize=(8, 6))

        # Plot the first distribution (assuming first half of data)
        # sns.scatterplot(x=X_tsne[:len(synth_ldp_X)//2, 0], y=X_tsne[:len(synth_ldp_X)//2, 1], color='blue', alpha=0.3, label='Original dataset')

        # Plot the second distribution (assuming second half of data)
        # sns.scatterplot(x=X_tsne[len(synth_ldp_X)//2:, 0], y=X_tsne[len(synth_ldp_X)//2:, 1], color='red', alpha=0.3, label='Synthetic dataset')

        # Add density plot for each distribution
        sns.kdeplot(x=X_tsne[:len(synth_ldp_X)//2, 0], y=X_tsne[:len(synth_ldp_X)//2, 1], color='blue', fill=True, cmap='Blues', alpha=0.4, label='Original dataset',bw_adjust=2.0)
        sns.kdeplot(x=X_tsne[len(synth_ldp_X)//2:, 0], y=X_tsne[len(synth_ldp_X)//2:, 1], color='red', fill=True, cmap='Reds', alpha=0.4, label='Synthetic dataset',bw_adjust=2.0)
        # Add density plot for each distribution
        
        # sns.kdeplot(X_tsne[:len(O_ldp_X)//2, 0], X_tsne[:len(O_ldp_X)//2, 1], color='blue', shade=True, cmap='Blues', label=None)
        # sns.kdeplot(X_tsne[len(O_ldp_X)//2:, 0], X_tsne[len(O_ldp_X)//2:, 1], color='red', shade=True, cmap='Reds', label=None)

        # Adjust labels and legensynth_grph_file_pathd
        plt.title('MST generated Synthetic data Vs. real data Distributions')
        # plt.xlabel('t-SNE Component 1')
        # plt.ylabel('t-SNE Component 2')
        plt.legend()
        ################# worked
        plt.savefig(synth_grph_file_path)
        plt.clf()
        ################# worked


        # # Initialize TSNE
        # tsne = TSNE(n_components=2, random_state=42)

        # # Fit and transform the data for the original dataset
        # X_tsne_o_ldp = tsne.fit_transform(O_ldp_X)
        # # X_tsne_updated = tsne.fit_transform(ldp_DS.X_train)

        # fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # # Plot the original dataset
        # plot_tsne(X_tsne_o_ldp, O_ldp_Y, 'LDP-vs_orig', axes[0])
        
        # O_Synth_X = np.vstack((X_train, Synth_DS.X_train))
        # O_Synth_Y = np.vstack((np.ones_like(DS.y_train), np.zeros_like(DS.y_train)))

        # X_tsne_o_synth = tsne.fit_transform(O_Synth_X)
        # X_tsne_o_synth = tsne.fit_transform(O_Synth_X)


        # plot_tsne(X_tsne_o_synth, O_Synth_Y, 'Synth-vs_orig', axes[1])

        
        # plt.show()
        # plt.savefig(grph_file_path)
        # plt.clf()

    else:
        if ( CF_method in ('NICE','ldp_ds_cf','synth_dp_cf','LDP_CF','LDP_SRR','LDP_Noisy_max') and n_count ==3 ) or CF_method in ('zerocost_DP_CF','inline_LDP'):
            
            CF_dir = './dpnice/cf_files/{}/'.format(dataset_name)
            # synth_DSoutdir = './dpnice/datasets_loaded/synth_{}/'.format(dataset_name)
            # modeloutdir = './dpnice/pretrained/{}/'.format(dataset_name)
            graphs_dir = './dpnice/cf_disributions/{}/{}/'.format(dataset_name,model)
            
            cf_file_name = '{}{}_{}_eps_{}_{}_k_{}.csv'.format(CF_dir, model, RANDOM_SEED,epsilon,CF_method,n_count)
            basic_cf_file_name = '{}{}_{}_eps_{}_{}_basic_k_{}.csv'.format(CF_dir, model, RANDOM_SEED,epsilon,CF_method,n_count)
            # DS_name = '{}{}_{}.pkl'.format(DSoutdir, model, RANDOM_SEED)
            # synth_DS_name = '{}{}_{}_{}.pkl'.format(synth_DSoutdir, model, RANDOM_SEED,epsilon)
            outputfile_name = '{}{}_{}_eps_{}_{}_k_{}.png'.format(graphs_dir, model, RANDOM_SEED,epsilon,CF_method,n_count)

            grph_file_path = '{}/SEED_{}_TSNE_eps_{}_ldp.png'.format(graphs_dir,RANDOM_SEED,epsilon) 
            title = '{}_{}_{}_eps:{}_{}_k:{}'.format(dataset_name,model,RANDOM_SEED,epsilon,CF_method,n_count)
            
            if not os.path.exists(graphs_dir):
                os.makedirs(graphs_dir)
            
            read_visualize_cfs(cf_file_name,basic_cf_file_name,outputfile_name,title)
    

# Create a figure with two subplots
    


    # O_Synth_X = np.vstack((X_train, Synth_DS.X_train))
    # O_Synth_Y = np.vstack((np.ones_like(DS.y_train), np.zeros_like(DS.y_train)))

    # plt.show()
    # plt.savefig(grph_file_path)
    # plt.clf()
# Plot the modified dataset
    # plot_tsne(X_tsne_updated, np.zeros_like(DS.y_train), 'ldp_set', axes[1])

    
# Display the plots
    