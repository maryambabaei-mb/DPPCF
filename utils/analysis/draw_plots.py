import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import pandas as pd
import os

def draw_CDF(input_array,epsilon=0,label="",marker = ".",variable = "epsilon"):
    sorted_input = np.sort(input_array)
    cdf = np.arange(1, len(sorted_input) + 1) / len(sorted_input)
    if(variable == "epsilon"):
        if epsilon != 0:
            line_label = str(label) + ",epsilon =" + str(epsilon)
        else:
            line_label = str(label)
    else:
        line_label = str(label) + ","+variable+" =" + str(epsilon)
    plt.plot(sorted_input, cdf, linestyle='-',label = line_label ) # marker=marker,
    

    # for i in range(1, len(sorted_input)):
    #     if sorted_input[i] != sorted_input[i-1]:
    #         plt.annotate(f'{cdf[i]:.2f}', xy=(0, cdf[i]), xytext=(-20, 0),
    #                  textcoords='offset points', ha='right', va='center', fontsize=8)
    ticks = np.linspace(min(sorted_input), max(sorted_input), num=5)  # Adjust the number of ticks as needed
    ticks = np.append(ticks, 1)  # Add 1 to the ticks
    plt.xticks(ticks)

    plt.xlabel('number of similar instances')
    plot_label = "fraction of results"
    plt.ylabel(plot_label)
    plt.title('Cumulative Distribution Function (CDF)')
    plt.grid(True)
    plt.legend()
    plt.close()



def draw_CDF_working(input_array,epsilon=0,label="",marker = ".",variable = "epsilon"):
    sorted_input = np.sort(input_array)
    cdf = np.arange(1, len(sorted_input) + 1) / len(sorted_input)
    if(variable == "epsilon"):
        if epsilon != 0:
            line_label = str(label) + ",epsilon =" + str(epsilon)
        else:
            line_label = str(label)
    else:
        line_label = str(label) + ","+variable+" =" + str(epsilon)
    plt.plot(sorted_input, cdf, linestyle='-',label = line_label ) # marker=marker,
    
    ticks = np.linspace(min(sorted_input), max(sorted_input), num=5)  # Adjust the number of ticks as needed
    ticks = np.append(ticks, 1)  # Add 1 to the ticks
    plt.xticks(ticks)

    plt.xlabel('number of similar instances')
    plot_label = "fraction of results"
    plt.ylabel(plot_label)
    plt.title('Cumulative Distribution Function (CDF)')
    plt.grid(True)
    plt.legend()
    plt.close()


def draw_dist(input_array,epsilon=0,label="",marker = "o",count= 100):
    x = np.arange(1, count+1, 1)
    if epsilon != 0:
        line_label = str(label) + ",epsilon =" + str(epsilon)
    else:
        line_label = str(label)
    plt.plot(x, input_array,linestyle='-', marker=marker,label = line_label )    #draw the line generated based on weight 
    plt.xlabel('counterfactual')
    plot_label = "distance"
    plt.ylabel(plot_label)
    plt.grid(True)
    plt.legend()
    plt.close()



def plot_number_frequency(value_list):
    # Count the frequency of each number
    frequency = Counter(numbers)
    
    # Extract numbers and corresponding frequencies
    numbers = list(frequency.keys())
    counts = list(frequency.values())
    
    # Plot the data
    plt.bar(numbers, counts, color='blue')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Statistical Analysis')
    plt.show()
    plt.close()


def draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_1, index_2 ,title,higher_better):
    # generate plot file path
    plot_file_path = '{}/{}/{}/'.format(graphs_dir,RANDOM_SEED,method_name)
    if not os.path.exists(plot_file_path):
        os.makedirs(plot_file_path, exist_ok=True)
    plot_file = '{}/{}.png'.format(plot_file_path,title)
    # draw plot 
    # param_1 = CF_DF.iloc[:,index1]
    # param_2 = CF_DF.iloc[:,index_2]
    # plt.figure(figsize=(14, 6))
    # sns.heatmap(pd.DataFrame({'param_1': param_1, 'param_2': param_2}), annot=True, cmap="viridis", cbar_kws={'label': 'Param 1 Value'}, fmt='d')
    # plt.title(title)
    param_1 = np.zeros(len(epsilons))
    param_2 = np.zeros(len(epsilons))
    for i in range(len(epsilons)):
        param_1[i] = CF_DF[i][index_1]
        param_2[i]= CF_DF[i][index_2]
    
    # Create a 2D array for heatmap where each row is the param value repeated
    # This is a workaround to use seaborn's heatmap for one-dimensional data
    combined_annotations = np.array([[f"{param_1[i]:.2f}\n{param_2[i]:.2f}"] for i in range(len(epsilons))])
    
    plt.figure(figsize=(6, 2))  # Adjusted figure size for one-dimensional data
    if higher_better:
        # sns.heatmap(param_1.reshape(-1,1), annot=combined_annotations, cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
        ax = sns.heatmap(param_1.reshape(1,-1), cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
    else:
        # sns.heatmap(param_1.reshape(-1,1), annot=combined_annotations, cmap="Blues",  fmt='', vmin=param_1.max(), vmax=param_1.min())
        ax = sns.heatmap(param_1.reshape(1,-1),  cmap="Blues",  fmt='', vmin=param_1.max(), vmax=param_1.min())

    for i in range(len(epsilons)):
    
        # Main value (param_1), larger and centered
        ax.text(i+.5,.5, f"{param_1[i]:.2f}", ha='center', va='center', fontsize=10)
        # Secondary value (param_2), smaller and in the top-right corner
        # This is a workaround; precise positioning like top-right is limited
        ax.text( i+0.7, .3,f"{param_2[i]:.2f}", ha='right', va='top', fontsize=6)


    # sns.heatmap(param_1, annot=combined_annotations, cmap="viridis", cbar_kws={'label': 'Param 1 Value'}, fmt='')
    # plt.title(f'{title} - Param')
    plt.xlabel('Epsilon values')
    plt.xticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons)
    
    plt.savefig(plot_file)
    plt.close()


def draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks ,index_1, index_2 ,title,higher_better):
    # generate plot file path
    plot_file_path = '{}/{}/{}/'.format(graphs_dir,RANDOM_SEED,method_name)
    if not os.path.exists(plot_file_path):
        os.makedirs(plot_file_path, exist_ok=True)
    plot_file = '{}/{}.png'.format(plot_file_path,title)
    
    # CF_DF is 2 dimensional array so, param_1 and param_2 are 2 dimensional arrays
    param_1 = np.zeros((len(epsilons), len(ks)))
    param_2 = np.zeros((len(epsilons), len(ks)))
    for i in range(len(epsilons)):
        for j in range(len(ks)):
            param_1[i][j] = CF_DF[i][j][index_1]
            param_2[i][j] = CF_DF[i][j][index_2]
    
    combined_annotations = np.array([[f"{param_1[i][j]:.2f}\n{param_2[i][j]:.2f}" for j in range(len(ks))] for i in range(len(epsilons))])

    plt.figure(figsize=(14, 6))
    if higher_better:
        # sns.heatmap(param_1, annot=combined_annotations, cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
        ax = sns.heatmap(param_1,  cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
    else:
        # sns.heatmap(param_1, annot=combined_annotations, cmap="Blues",  fmt='', vmin=param_1.max(), vmax=param_1.min())
        ax = sns.heatmap(param_1, cmap="Blues",  fmt='', vmin=param_1.max(), vmax=param_1.min())
    
    for i in range(len(epsilons)):
        for j in range(len(ks)):
            # Main value (param_1), larger and centered
            ax.text(j+.5, i+.5, f"{param_1[i][j]:.2f}", ha='center', va='center', fontsize=12)
            # Secondary value (param_2), smaller and in the top-right corner
            # This is a workaround; precise positioning like top-right is limited
            ax.text(j+0.7, i+0.3, f"{param_2[i][j]:.2f}", ha='right', va='top', fontsize=9)


    # plt.title(f'{title} - Param 1 & Param 2')
    plt.xlabel('k values')
    plt.ylabel('Epsilon values')
    plt.xticks(ticks=np.arange(len(ks)) + 0.5, labels=ks)
    plt.yticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons, rotation=0)

    plt.savefig(plot_file)
    plt.close()


    ########################################################################
    # 3 param heatmaps                                              ########
    ########################################################################
def draw_one_dimensional_heatmap_3param(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_1, index_2 , index_3 ,nice_param,title,higher_better):
    # generate plot file path
    if RANDOM_SEED != 9999:  ### aggregated results
        plot_file_path = '{}/{}/{}/'.format(graphs_dir,RANDOM_SEED,method_name)
    else:
        plot_file_path = '{}/{}/'.format(graphs_dir,method_name)
    if not os.path.exists(plot_file_path):
        os.makedirs(plot_file_path, exist_ok=True)
    plot_file = '{}/{}.png'.format(plot_file_path,title)
    imp_plot_file = '{}/{}_improvement.png'.format(plot_file_path,title)

    # draw plot 
    # param_1 = CF_DF.iloc[:,index1]
    # param_2 = CF_DF.iloc[:,index_2]
    # plt.figure(figsize=(14, 6))
    # sns.heatmap(pd.DataFrame({'param_1': param_1, 'param_2': param_2}), annot=True, cmap="viridis", cbar_kws={'label': 'Param 1 Value'}, fmt='d')
    # plt.title(title)
    param_1 = np.zeros(len(epsilons))
    param_2 = np.zeros(len(epsilons))
    param_3 = np.zeros(len(epsilons))
    for i in range(len(epsilons)):
        param_1[i] = CF_DF[i][index_1]
        param_2[i]= CF_DF[i][index_2]
        param_3[i]= CF_DF[i][index_3]
    # Create a 2D array for heatmap where each row is the param value repeated
    # This is a workaround to use seaborn's heatmap for one-dimensional data
    # combined_annotations = np.array([[f"{param_1[i]:.2f}\n{param_2[i]:.2f}"] for i in range(len(epsilons))])
    
    plt.figure(figsize=(8, 3))  # Adjusted figure size for one-dimensional data
    if index_1 in higher_better:
        # sns.heatmap(param_1.reshape(-1,1), annot=combined_annotations, cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
        ax = sns.heatmap(param_1.reshape(1,-1), cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
    else:
        # sns.heatmap(param_1.reshape(-1,1), annot=combined_annotations, cmap="Blues",  fmt='', vmin=param_1.max(), vmax=param_1.min())
        ax = sns.heatmap(param_1.reshape(1,-1),  cmap="Blues",  fmt='', vmin=param_1.max(), vmax=param_1.min())

    for i in range(len(epsilons)):
    
        # Main value (param_1), larger and centered
        ax.text(i+.4,.5, f"{param_1[i]:.2f}", ha='center', va='center', fontsize=10)
        # Secondary value (param_2), smaller and in the top-right corner
        # This is a workaround; precise positioning like top-right is limited
        ax.text( i+0.8, .3,f"{param_2[i]:.2f}", ha='right', va='top', fontsize=6)
        ax.text( i+0.8, .7,f"{param_3[i]:.2f}", ha='right', va='bottom', fontsize=6)


    # sns.heatmap(param_1, annot=combined_annotations, cmap="viridis", cbar_kws={'label': 'Param 1 Value'}, fmt='')
    # plt.title(f'{title} - Param')
    mytitle = 'NICE:priv:{}/distance:{}/plaus_distance:{}/'.format(nice_param[index_1],nice_param[index_2],nice_param[index_3])
    plt.title(mytitle)
    plt.xlabel('Epsilon values')
    plt.xticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons)
    plt.savefig(plot_file)
    plt.close()
    
    ########################################################################
    ############ Draw improvement heatmaps ############################
    ########################################################################
    imp_1 = np.zeros(len(epsilons))
    imp_2 = np.zeros(len(epsilons))
    imp_3 = np.zeros(len(epsilons))
    for i in range(len(epsilons)):
        if index_1 in higher_better:
            imp_1[i] = (param_1[i] - nice_param[index_1]) / nice_param[index_1]
        else:
            imp_1[i] = (nice_param[index_1] - param_1[i]) / nice_param[index_1]
        if index_2 in higher_better:
            imp_2[i] = (param_2[i] - nice_param[index_2]) / nice_param[index_2]
        else:
            imp_2[i] = (nice_param[index_2] - param_2[i]) / nice_param[index_2]
        if index_3 in higher_better:
            imp_3[i] = (param_3[i] - nice_param[index_3]) / nice_param[index_3]
        else:
            imp_3[i] = (nice_param[index_3] - param_3[i]) / nice_param[index_3]


    plt.figure(figsize=(6, 2))  # Adjusted figure size for one-dimensional data
        # sns.heatmap(param_1.reshape(-1,1), annot=combined_annotations, cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
    ax = sns.heatmap(imp_1.reshape(1,-1), cmap="Purples",  fmt='', vmin=imp_1.min(), vmax=imp_1.max())
    
    for i in range(len(epsilons)):
    
        # Main value (param_1), larger and centered
        ax.text(i+.4,.5, f"{imp_1[i]:.2f}", ha='center', va='center', fontsize=10)
        # Secondary value (param_2), smaller and in the top-right corner
        # This is a workaround; precise positioning like top-right is limited
        ax.text( i+0.8, .3,f"{imp_2[i]:.2f}", ha='right', va='top', fontsize=6)
        ax.text( i+0.8, .7,f"{imp_3[i]:.2f}", ha='right', va='bottom', fontsize=6)


    # sns.heatmap(param_1, annot=combined_annotations, cmap="viridis", cbar_kws={'label': 'Param 1 Value'}, fmt='')
    # plt.title(f'{title} - Param')
    plt.xlabel('Epsilon values')
    plt.xticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons)
    
    plt.savefig(imp_plot_file)
    plt.close()

    

def draw_two_dimensional_heatmap_3param(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks ,index_1, index_2 , index_3 ,nice_param,title,higher_better):
    # generate plot file path
    if RANDOM_SEED != 9999:  ### aggregated results
        plot_file_path = '{}/{}/{}/'.format(graphs_dir,RANDOM_SEED,method_name)
    else:
        plot_file_path = '{}/{}/'.format(graphs_dir,method_name)

    if not os.path.exists(plot_file_path):
        os.makedirs(plot_file_path, exist_ok=True)
    plot_file = '{}/{}.png'.format(plot_file_path,title)
    imp_plot_file = '{}/{}_improvement.png'.format(plot_file_path,title)
    # CF_DF is 2 dimensional array so, param_1 and param_2 are 2 dimensional arrays
    # param_1 = np.zeros((len(epsilons), len(ks)))
    # param_2 = np.zeros((len(epsilons), len(ks)))
    # param_3 = np.zeros((len(epsilons), len(ks)))
    # for i in range(len(epsilons)):
    #     for j in range(len(ks)):
    #         param_1[i][j] = CF_DF[i][j][index_1]
    #         param_2[i][j] = CF_DF[i][j][index_2]
    #         param_3[i][j] = CF_DF[i][j][index_3]
    param_1 = np.zeros((len(ks),len(epsilons)))
    param_2 = np.zeros((len(ks),len(epsilons)))
    param_3 = np.zeros((len(ks),len(epsilons)))
    for i in range(len(ks)):
        for j in range(len(epsilons)):
            param_1[i][j] = CF_DF[j][i][index_1]
            param_2[i][j] = CF_DF[j][i][index_2]
            param_3[i][j] = CF_DF[j][i][index_3]
    # combined_annotations = np.array([[f"{param_1[i][j]:.2f}\n{param_2[i][j]:.2f}" for j in range(len(ks))] for i in range(len(epsilons))])

    plt.figure(figsize=(16, 7))
    if index_1 in higher_better:
        # sns.heatmap(param_1, annot=combined_annotations, cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
        ax = sns.heatmap(param_1,  cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
    else:
        # sns.heatmap(param_1, annot=combined_annotations, cmap="Blues",  fmt='', vmin=param_1.max(), vmax=param_1.min())
        ax = sns.heatmap(param_1, cmap="Blues",  fmt='', vmin=param_1.max(), vmax=param_1.min())
    
    # for i in range(len(epsilons)):
    #     for j in range(len(ks)):
    #         # Main value (param_1), larger and centered
    #         ax.text(j+.4, i+.5, f"{param_1[i][j]:.2f}", ha='center', va='center', fontsize=12)
    #         # Secondary value (param_2), smaller and in the top-right corner
    #         # This is a workaround; precise positioning like top-right is limited
    #         ax.text(j+0.8, i+0.3, f"{param_2[i][j]:.2f}", ha='right', va='top', fontsize=9)
    #         ax.text(j+0.8, i+0.7, f"{param_3[i][j]:.2f}", ha='right', va='top', fontsize=9)

    for i in range(len(ks)):
        for j in range(len(epsilons)):
            # Main value (param_1), larger and centered
            ax.text(j+.4, i+.5, f"{param_1[i][j]:.2f}", ha='center', va='center', fontsize=12)
            # Secondary value (param_2), smaller and in the top-right corner
            # This is a workaround; precise positioning like top-right is limited
            ax.text(j+0.8, i+0.3, f"{param_2[i][j]:.2f}", ha='right', va='top', fontsize=9)
            ax.text(j+0.8, i+0.7, f"{param_3[i][j]:.2f}", ha='right', va='top', fontsize=9)


    # plt.title(f'{title} - Param 1 & Param 2')
    mytitle = 'NICE:priv:{}/distance:{}/plaus_distance:{}/'.format(nice_param[index_1],nice_param[index_2],nice_param[index_3])
    plt.title(mytitle)
    plt.xlabel('Elsilon')
    plt.ylabel('k')
    plt.xticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons)
    plt.yticks(ticks=np.arange(len(ks)) + 0.5, labels=ks, rotation=0)
    
    plt.savefig(plot_file)
    plt.close()

        ########################################################################
    ############ Draw improvement heatmaps ############################
    ########################################################################
    imp_1 =  np.zeros((len(ks),len(epsilons)))
    imp_2 =  np.zeros((len(ks),len(epsilons)))
    imp_3 =  np.zeros((len(ks),len(epsilons)))
    for i in range(len(ks)):
        for j in range(len(epsilons)):
            if index_1 in higher_better:
                imp_1[i][j] = (param_1[i][j] - nice_param[index_1]) / nice_param[index_1]
            else:
                imp_1[i][j] = (nice_param[index_1] - param_1[i][j]) / nice_param[index_1]
            if index_2 in higher_better:
                imp_2[i][j] = (param_2[i][j] - nice_param[index_2]) / nice_param[index_2]
            else:
                imp_2[i][j] = (nice_param[index_2] - param_2[i][j]) / nice_param[index_2]
            if index_3 in higher_better:
                imp_3[i][j] = (param_3[i][j] - nice_param[index_3]) / nice_param[index_3]
            else:
                imp_3[i][j] = (nice_param[index_3] - param_3[i][j]) / nice_param[index_3]


    plt.figure(figsize=(14, 6))
       
    ax = sns.heatmap(param_1,  cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
    
    for i in range(len(ks)):
        for j in range(len(epsilons)):
            # Main value (param_1), larger and centered
            ax.text(j+.4, i+.5, f"{imp_1[i][j]}", ha='center', va='center', fontsize=12)
            # Secondary value (param_2), smaller and in the top-right corner
            # This is a workaround; precise positioning like top-right is limited
            ax.text(j+0.8, i+0.3, f"{imp_2[i][j]}", ha='right', va='top', fontsize=9)
            ax.text(j+0.8, i+0.7, f"{imp_3[i][j]}", ha='right', va='top', fontsize=9)



    # sns.heatmap(param_1, annot=combined_annotations, cmap="viridis", cbar_kws={'label': 'Param 1 Value'}, fmt='')
    # plt.title(f'{title} - Param')
    plt.xlabel('Epsilon values')
    plt.xticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons)
    
    plt.savefig(imp_plot_file)
    plt.close()


def draw_one_dimensional_heatmap_4param(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_1, index_2 , index_3 ,index_4,nice_param,title,higher_better):
    # generate plot file path
    if RANDOM_SEED != 9999:  ### aggregated results
        plot_file_path = '{}/{}/{}/'.format(graphs_dir,RANDOM_SEED,method_name)
    else:
        plot_file_path = '{}/{}/'.format(graphs_dir,method_name)
    if not os.path.exists(plot_file_path):
        os.makedirs(plot_file_path, exist_ok=True)
    plot_file = '{}/{}.png'.format(plot_file_path,title)
    imp_plot_file = '{}/{}_improvement.png'.format(plot_file_path,title)

    
    param_1 = np.zeros(len(epsilons))
    param_2 = np.zeros(len(epsilons))
    param_3 = np.zeros(len(epsilons))
    param_4 = np.zeros(len(epsilons))
    for i in range(len(epsilons)):
        param_1[i] = CF_DF[i][index_1]
        param_2[i]= CF_DF[i][index_2]
        param_3[i]= CF_DF[i][index_3]
        param_4[i]= CF_DF[i][index_4]
    # Create a 2D array for heatmap where each row is the param value repeated
    # This is a workaround to use seaborn's heatmap for one-dimensional data
    # combined_annotations = np.array([[f"{param_1[i]:.2f}\n{param_2[i]:.2f}"] for i in range(len(epsilons))])
    
    plt.figure(figsize=(8, 2))  # Adjusted figure size for one-dimensional data
    cbar_title = {'label': f'{title}'}
    purples_palette = sns.light_palette("purple", as_cmap=True)
    blues_palette = sns.light_palette("blue", as_cmap=True)
    if index_1 in higher_better:
        # sns.heatmap(param_1.reshape(-1,1), annot=combined_annotations, cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
        ax = sns.heatmap(param_1.reshape(1,-1), cmap=purples_palette,  fmt='', vmin=param_1.min(), vmax=param_1.max(),cbar_kws=cbar_title)
    else:
        # sns.heatmap(param_1.reshape(-1,1), annot=combined_annotations, cmap="Blues",  fmt='', vmin=param_1.max(), vmax=param_1.min())
        ax = sns.heatmap(param_1.reshape(1,-1),  cmap=blues_palette,  fmt='', vmin=param_1.max(), vmax=param_1.min(),cbar_kws=cbar_title)

    for i in range(len(epsilons)):
    
        # Main value (param_1), larger and centered
        ax.text(i+.3,.5, f"{param_1[i]:.2f}", ha='center', va='center', fontsize=11)
        # Secondary value (param_2), smaller and in the top-right corner
        # This is a workaround; precise positioning like top-right is limited
        ax.text( i+0.8, .3,f"{param_2[i]:.2f}", ha='right', va='top', fontsize=7)
        ax.text( i+0.8, .7,f"{param_3[i]:.2f}", ha='right', va='top', fontsize=7)
        ax.text( i+0.2, .7,f"{param_4[i]:.2f}", ha='right', va='top', fontsize=7)


    # plt.title(f'{title} - Param')
    mytitle = 'NICE:priv:{}/distance:{}/plaus_distance:{}/correctness:{}'.format(nice_param[index_1],nice_param[index_2],nice_param[index_3],nice_param[index_4])
    plt.title(mytitle)
    plt.xlabel('Epsilon values')
    plt.xticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons)
    plt.savefig(plot_file)
    plt.close()
    
    ########################################################################
    ############ Draw improvement heatmaps ############################
    ########################################################################
    imp_1 = np.zeros(len(epsilons))
    imp_2 = np.zeros(len(epsilons))
    imp_3 = np.zeros(len(epsilons))
    imp_4 = np.zeros(len(epsilons))
    for i in range(len(epsilons)):
        if index_1 in higher_better:
            imp_1[i] = ((param_1[i] - nice_param[index_1]) / nice_param[index_1])*100
        else:
            imp_1[i] = ((nice_param[index_1] - param_1[i]) / nice_param[index_1])*100
        if index_2 in higher_better:
            imp_2[i] = ((param_2[i] - nice_param[index_2]) / nice_param[index_2])*100
        else:
            imp_2[i] = ((nice_param[index_2] - param_2[i]) / nice_param[index_2])*100
        if index_3 in higher_better:
            imp_3[i] = ((param_3[i] - nice_param[index_3]) / nice_param[index_3])*100
        else:
            imp_3[i] = ((nice_param[index_3] - param_3[i]) / nice_param[index_3])*100
        if index_4 in higher_better:
            imp_4[i] = ((param_4[i] - nice_param[index_4]) / nice_param[index_4])*100
        else:
            imp_4[i] = ((nice_param[index_4] - param_4[i]) / nice_param[index_4])*100


    plt.figure(figsize=(7, 2))  # Adjusted figure size for one-dimensional data
        # sns.heatmap(param_1.reshape(-1,1), annot=combined_annotations, cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
    cbar_title = {'label': f'{title}  improvement'} 
    norm = plt.Normalize(vmin=imp_1.min(), vmax=imp_1.max())
    purples_palette = sns.light_palette("purple", as_cmap=True)
    blues_palette = sns.light_palette("blue", as_cmap=True)
    ax = sns.heatmap(imp_1.reshape(1,-1), cmap=purples_palette,  fmt='', norm = norm,cbar_kws=cbar_title)
    
    for i in range(len(epsilons)):
    
        # Main value (param_1), larger and centered
        ax.text(i+.5,.5, f"{imp_1[i]:.2f}", ha='center', va='center', fontsize=10)
        # Secondary value (param_2), smaller and in the top-right corner
        ax.text( i+0.8, .3,f"{imp_2[i]:.1f}", ha='right', va='top', fontsize=7)
        ax.text( i+0.8, .7,f"{imp_3[i]:.1f}", ha='right', va='top', fontsize=7)
        ax.text( i+0.3, .7,f"{imp_4[i]:.1f}", ha='right', va='top', fontsize=7)


    # sns.heatmap(param_1, annot=combined_annotations, cmap="viridis", cbar_kws={'label': 'Param 1 Value'}, fmt='')
    # plt.title(f'{title} - Param')
    plt.cbar_kws={'label': f'{title} improvement'}
    plt.xlabel('Epsilon values')
    plt.xticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons)
    
    plt.savefig(imp_plot_file)
    plt.close()

    
def draw_two_dimensional_heatmap_4param(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks ,index_1, index_2 , index_3 ,index_4,nice_param,title,higher_better):
    # generate plot file path
    if RANDOM_SEED != 9999:  ### aggregated results
        plot_file_path = '{}/{}/{}/'.format(graphs_dir,RANDOM_SEED,method_name)
    else:
        plot_file_path = '{}/{}/'.format(graphs_dir,method_name)

    if not os.path.exists(plot_file_path):
        os.makedirs(plot_file_path, exist_ok=True)
    plot_file = '{}/{}.png'.format(plot_file_path,title)
    imp_plot_file = '{}/{}_improvement.png'.format(plot_file_path,title)
    # CF_DF is 2 dimensional array so, param_1 and param_2 are 2 dimensional arrays
    # param_1 = np.zeros((len(epsilons), len(ks)))
    # param_2 = np.zeros((len(epsilons), len(ks)))
    # param_3 = np.zeros((len(epsilons), len(ks)))
    # for i in range(len(epsilons)):
    #     for j in range(len(ks)):
    #         param_1[i][j] = CF_DF[i][j][index_1]
    #         param_2[i][j] = CF_DF[i][j][index_2]
    #         param_3[i][j] = CF_DF[i][j][index_3]
    param_1 = np.zeros((len(ks),len(epsilons)))
    param_2 = np.zeros((len(ks),len(epsilons)))
    param_3 = np.zeros((len(ks),len(epsilons)))
    param_4 = np.zeros((len(ks),len(epsilons)))
    for i in range(len(ks)):
        for j in range(len(epsilons)):
            param_1[i][j] = CF_DF[j][i][index_1]
            param_2[i][j] = CF_DF[j][i][index_2]
            param_3[i][j] = CF_DF[j][i][index_3]
            param_4[i][j] = CF_DF[j][i][index_4]
    # combined_annotations = np.array([[f"{param_1[i][j]:.2f}\n{param_2[i][j]:.2f}" for j in range(len(ks))] for i in range(len(epsilons))])

    plt.figure(figsize=(16, 7))
    cbar_title = {'label': f'{title}'}
    purples_palette = sns.light_palette("purple", as_cmap=True)
    blue_palette = sns.light_palette("blue", as_cmap=True)
    if index_1 in higher_better:
        # sns.heatmap(param_1, annot=combined_annotations, cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
        ax = sns.heatmap(param_1,  cmap=purples_palette,  fmt='', vmin=param_1.min(), vmax=param_1.max(),cbar_kws=cbar_title)
    else:
        # sns.heatmap(param_1, annot=combined_annotations, cmap="Blues",  fmt='', vmin=param_1.max(), vmax=param_1.min())
        ax = sns.heatmap(param_1, cmap=blue_palette,  fmt='', vmin=param_1.max(), vmax=param_1.min(),cbar_kws=cbar_title)
    
    
    for i in range(len(ks)):
        for j in range(len(epsilons)):
            # Main value (param_1), larger and centered
            ax.text(j+.5, i+.5, f"{param_1[i][j]:.2f}", ha='center', va='center', fontsize=12)
            # Secondary value (param_2), smaller and in the top-right corner
            # This is a workaround; precise positioning like top-right is limited
            ax.text(j+0.8, i+0.3, f"{param_2[i][j]:.2f}", ha='right', va='top', fontsize=9)
            ax.text(j+0.8, i+0.7, f"{param_3[i][j]:.2f}", ha='right', va='top', fontsize=9)
            ax.text(j+0.3, i+0.7, f"{param_4[i][j]:.2f}", ha='right', va='top', fontsize=9)


    # plt.title(f'{title} - Param 1 & Param 2')
    
    mytitle = 'NICE:priv:{}/distance:{}/plaus_distance:{}/correctness:{}'.format(nice_param[index_1],nice_param[index_2],nice_param[index_3],nice_param[index_4])
    plt.title(mytitle)
    plt.xlabel('Elsilon')
    plt.ylabel('k')
    plt.xticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons)
    plt.yticks(ticks=np.arange(len(ks)) + 0.5, labels=ks, rotation=0)
    
    plt.savefig(plot_file)
    plt.close()

        ########################################################################
    ############ Draw improvement heatmaps ############################
    ########################################################################
    imp_1 =  np.zeros((len(ks),len(epsilons)))
    imp_2 =  np.zeros((len(ks),len(epsilons)))
    imp_3 =  np.zeros((len(ks),len(epsilons)))
    imp_4 =  np.zeros((len(ks),len(epsilons)))
    for i in range(len(ks)):
        for j in range(len(epsilons)):
            if index_1 in higher_better:
                imp_1[i][j] = ((param_1[i][j] - nice_param[index_1]) / nice_param[index_1])*100
            else:
                imp_1[i][j] = ((nice_param[index_1] - param_1[i][j]) / nice_param[index_1])*100
            if index_2 in higher_better:
                imp_2[i][j] = ((param_2[i][j] - nice_param[index_2]) / nice_param[index_2])*100
            else:
                imp_2[i][j] = ((nice_param[index_2] - param_2[i][j]) / nice_param[index_2])*100
            if index_3 in higher_better:
                imp_3[i][j] = ((param_3[i][j] - nice_param[index_3]) / nice_param[index_3])*100
            else:
                imp_3[i][j] = ((nice_param[index_3] - param_3[i][j]) / nice_param[index_3])*100
            if index_4 in higher_better:
                imp_4[i][j] = ((param_4[i][j] - nice_param[index_4]) / nice_param[index_4])*100
            else:
                imp_4[i][j] = ((nice_param[index_4] - param_4[i][j]) / nice_param[index_4])*100


    plt.figure(figsize=(15,5))
    cbar_title = {'label': f'{title}  improvement'} 
    norm = plt.Normalize(vmin=imp_1.min(), vmax=imp_1.max())
    purples_palette = sns.light_palette("purple", as_cmap=True)
    blue_palette = sns.light_palette("blue", as_cmap=True)
    ax = sns.heatmap(imp_1,  cmap=blue_palette,  fmt='', norm = norm,cbar_kws=cbar_title)
    
    for i in range(len(ks)):
        for j in range(len(epsilons)):
            # Main value (param_1), larger and centered
            ax.text(j+.5, i+.5, f"{imp_1[i][j]:.2f}", ha='center', va='center', fontsize=11)
            # Secondary value (param_2), smaller and in the corners
            ax.text(j+0.8, i+0.2, f"{imp_2[i][j]:.1f}", ha='right', va='top', fontsize=11)
            ax.text(j+0.8, i+0.8, f"{imp_3[i][j]:.1f}", ha='right', va='top', fontsize=11)
            ax.text(j+0.2, i+0.8, f"{imp_4[i][j]:.1f}", ha='right', va='top', fontsize=11)



    
    # plt.title={'label': f'{title} improvement'}
    plt.xlabel('Epsilon values')
    plt.xticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons)
    plt.yticks(ticks=np.arange(len(ks)) + 0.5, labels=ks, rotation=0)
    plt.savefig(imp_plot_file)
    plt.close()

def draw_one_dimensional_heatmap_5param(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_1, index_2 , index_3 ,index_4,index_5,nice_param,title,higher_better):
    # generate plot file path
    if RANDOM_SEED != 9999:  ### aggregated results
        plot_file_path = '{}/{}/{}/'.format(graphs_dir,RANDOM_SEED,method_name)
    else:
        plot_file_path = '{}/{}/'.format(graphs_dir,method_name)
    if not os.path.exists(plot_file_path):
        os.makedirs(plot_file_path, exist_ok=True)
    plot_file = '{}/{}.png'.format(plot_file_path,title)
    imp_plot_file = '{}/{}_improvement.png'.format(plot_file_path,title)

    
    param_1 = np.zeros(len(epsilons))
    param_2 = np.zeros(len(epsilons))
    param_3 = np.zeros(len(epsilons))
    param_4 = np.zeros(len(epsilons))
    param_5 = np.zeros(len(epsilons))
    for i in range(len(epsilons)):
        param_1[i] = CF_DF[i][index_1]
        param_2[i]= CF_DF[i][index_2]
        param_3[i]= CF_DF[i][index_3]
        param_4[i]= CF_DF[i][index_4]
        param_5[i]= CF_DF[i][index_5]
    # Create a 2D array for heatmap where each row is the param value repeated
    # This is a workaround to use seaborn's heatmap for one-dimensional data
    # combined_annotations = np.array([[f"{param_1[i]:.2f}\n{param_2[i]:.2f}"] for i in range(len(epsilons))])
    
    plt.figure(figsize=(8, 2))  # Adjusted figure size for one-dimensional data
    cbar_title = {'label': f'{title}'}
    purples_palette = sns.light_palette("purple", as_cmap=True)
    blues_palette = sns.light_palette("blue", as_cmap=True)
    if index_1 in higher_better:
        # sns.heatmap(param_1.reshape(-1,1), annot=combined_annotations, cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
        ax = sns.heatmap(param_1.reshape(1,-1), cmap=purples_palette,  fmt='', vmin=param_1.min(), vmax=param_1.max(),cbar_kws=cbar_title)
    else:
        # sns.heatmap(param_1.reshape(-1,1), annot=combined_annotations, cmap="Blues",  fmt='', vmin=param_1.max(), vmax=param_1.min())
        ax = sns.heatmap(param_1.reshape(1,-1),  cmap=blues_palette,  fmt='', vmin=param_1.max(), vmax=param_1.min(),cbar_kws=cbar_title)

    for i in range(len(epsilons)):
    
        # Main value (param_1), larger and centered
        ax.text(i+.3,.5, f"{param_1[i]:.2f}", ha='center', va='center', fontsize=11)
        # Secondary value (param_2), smaller and in the top-right corner
        # This is a workaround; precise positioning like top-right is limited
        ax.text( i+0.8, .3,f"{param_2[i]:.2f}", ha='right', va='top', fontsize=7)
        ax.text( i+0.8, .8,f"{param_3[i]:.2f}", ha='right', va='top', fontsize=7)
        ax.text( i+0.3, .8,f"{param_4[i]:.2f}", ha='right', va='top', fontsize=7)
        ax.text( i+0.8, .8,f"{param_5[i]:.2f}", ha='right', va='top', fontsize=7)


    # plt.title(f'{title} - Param')
    mytitle = 'NICE:priv:{}/distance:{}/plaus_distance:{}/correctness:{}'.format(nice_param[index_1],nice_param[index_2],nice_param[index_3],nice_param[index_4])
    plt.title(mytitle)
    plt.xlabel('Epsilon values')
    plt.xticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons)
    plt.savefig(plot_file)
    plt.close()
    
    ########################################################################
    ############ Draw improvement heatmaps ############################
    ########################################################################
    imp_1 = np.zeros(len(epsilons))
    imp_2 = np.zeros(len(epsilons))
    imp_3 = np.zeros(len(epsilons))
    imp_4 = np.zeros(len(epsilons))
    imp_5 = np.zeros(len(epsilons))
    for i in range(len(epsilons)):
        if index_1 in higher_better:
            imp_1[i] = ((param_1[i] - nice_param[index_1]) / ((nice_param[index_1])+0.000000000001))*100
        else:
            imp_1[i] = ((nice_param[index_1] - param_1[i]) / ((nice_param[index_1])+0.000000000001))*100
        if index_2 in higher_better:
            imp_2[i] = ((param_2[i] - nice_param[index_2]) / ((nice_param[index_2])+0.000000000001))*100
        else:
            imp_2[i] = ((nice_param[index_2] - param_2[i]) / ((nice_param[index_2])+0.000000000001))*100
        if index_3 in higher_better:
            imp_3[i] = ((param_3[i] - nice_param[index_3]) / ((nice_param[index_3])+0.000000000001))*100
        else:
            imp_3[i] = ((nice_param[index_3] - param_3[i]) / ((nice_param[index_3])+0.000000000001))*100
        if index_4 in higher_better:
            imp_4[i] = ((param_4[i] - nice_param[index_4]) / ((nice_param[index_4])+0.000000000001))*100
        else:
            imp_4[i] = ((nice_param[index_4] - param_4[i]) / ((nice_param[index_4])+0.000000000001))*100
        if index_5 in higher_better:
            imp_5[i] = ((param_5[i] - nice_param[index_5]) / ((nice_param[index_5])+0.000000000001))*100
        else:
            imp_5[i] = ((nice_param[index_5] - param_5[i]) / ((nice_param[index_5])+0.000000000001))*100

    plt.figure(figsize=(10, 2))  # Adjusted figure size for one-dimensional data
        # sns.heatmap(param_1.reshape(-1,1), annot=combined_annotations, cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
    cbar_title = {'label': f'{title}  improvement'} 
    norm = plt.Normalize(vmin=imp_1.min(), vmax=imp_1.max())
    purples_palette = sns.light_palette("purple", as_cmap=True)
    blues_palette = sns.light_palette("blue", as_cmap=True)
    ax = sns.heatmap(imp_1.reshape(1,-1), cmap=purples_palette,  fmt='', norm = norm,cbar_kws=cbar_title)
    def get_text_color(background_color):
        r, g, b, _ = background_color
        luminance = 0.299*r + 0.587*g + 0.114*b
        return 'white' if luminance < 0.5 else 'black'
    for i in range(len(epsilons)):
        background_color = purples_palette(norm(imp_1[i]))
        text_color = get_text_color(background_color)
        # Main value (param_1), larger and centered
        ax.text(i+.5,.5, f"{imp_1[i]:.2f}", ha='center', va='center', fontsize=11,color=text_color)
        # Secondary value (param_2), smaller and in the top-right corner
        ax.text( i+0.9, .2,f"{imp_2[i]:.1f}", ha='right', va='top', fontsize=11,color=text_color)
        ax.text( i+0.9, .7,f"{imp_3[i]:.1f}", ha='right', va='top', fontsize=11,color=text_color)
        ax.text( i+0.4, .7,f"{imp_4[i]:.1f}", ha='right', va='top', fontsize=11,color=text_color)
        ax.text( i+0.4, .2,f"{imp_5[i]:.1f}", ha='right', va='top', fontsize=11,color=text_color)


    # sns.heatmap(param_1, annot=combined_annotations, cmap="viridis", cbar_kws={'label': 'Param 1 Value'}, fmt='')
    # plt.title(f'{title} - Param')
    plt.cbar_kws={'label': f'{title} improvement'}
    plt.xlabel('Epsilon values')
    plt.xticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons)
    plt.savefig(imp_plot_file)
    plt.close()


def draw_two_dimensional_heatmap_5param(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks ,index_1, index_2 , index_3 ,index_4,index_5,nice_param,title,higher_better):
    # generate plot file path
    if RANDOM_SEED != 9999:  ### aggregated results
        plot_file_path = '{}/{}/{}/'.format(graphs_dir,RANDOM_SEED,method_name)
    else:
        plot_file_path = '{}/{}/'.format(graphs_dir,method_name)

    if not os.path.exists(plot_file_path):
        os.makedirs(plot_file_path, exist_ok=True)
    plot_file = '{}/{}.png'.format(plot_file_path,title)
    imp_plot_file = '{}/{}_improvement.png'.format(plot_file_path,title)
    # CF_DF is 2 dimensional array so, param_1 and param_2 are 2 dimensional arrays
    # param_1 = np.zeros((len(epsilons), len(ks)))
    # param_2 = np.zeros((len(epsilons), len(ks)))
    # param_3 = np.zeros((len(epsilons), len(ks)))
    # for i in range(len(epsilons)):
    #     for j in range(len(ks)):
    #         param_1[i][j] = CF_DF[i][j][index_1]
    #         param_2[i][j] = CF_DF[i][j][index_2]
    #         param_3[i][j] = CF_DF[i][j][index_3]
    param_1 = np.zeros((len(ks),len(epsilons)))
    param_2 = np.zeros((len(ks),len(epsilons)))
    param_3 = np.zeros((len(ks),len(epsilons)))
    param_4 = np.zeros((len(ks),len(epsilons)))
    param_5 = np.zeros((len(ks),len(epsilons)))
    for i in range(len(ks)):
        for j in range(len(epsilons)):
            param_1[i][j] = CF_DF[j][i][index_1]
            param_2[i][j] = CF_DF[j][i][index_2]
            param_3[i][j] = CF_DF[j][i][index_3]
            param_4[i][j] = CF_DF[j][i][index_4]
            param_5[i][j] = CF_DF[j][i][index_5]
    # combined_annotations = np.array([[f"{param_1[i][j]:.2f}\n{param_2[i][j]:.2f}" for j in range(len(ks))] for i in range(len(epsilons))])

    plt.figure(figsize=(16, 7))
    cbar_title = {'label': f'{title}'}
    purples_palette = sns.light_palette("purple", as_cmap=True)
    blue_palette = sns.light_palette("blue", as_cmap=True)
    if index_1 in higher_better:
        # sns.heatmap(param_1, annot=combined_annotations, cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
        ax = sns.heatmap(param_1,  cmap=purples_palette,  fmt='', vmin=param_1.min(), vmax=param_1.max(),cbar_kws=cbar_title)
    else:
        # sns.heatmap(param_1, annot=combined_annotations, cmap="Blues",  fmt='', vmin=param_1.max(), vmax=param_1.min())
        ax = sns.heatmap(param_1, cmap=blue_palette,  fmt='', vmin=param_1.max(), vmax=param_1.min(),cbar_kws=cbar_title)
    
    
    for i in range(len(ks)):
        for j in range(len(epsilons)):
            # Main value (param_1), larger and centered
            ax.text(j+.5, i+.5, f"{param_1[i][j]:.2f}", ha='center', va='center', fontsize=11)
            # Secondary value (param_2), smaller and in the top-right corner
            # This is a workaround; precise positioning like top-right is limited
            ax.text(j+0.8, i+0.2, f"{param_2[i][j]:.2f}", ha='right', va='top', fontsize=11)
            ax.text(j+0.8, i+0.8, f"{param_3[i][j]:.2f}", ha='right', va='top', fontsize=11)
            ax.text(j+0.2, i+0.8, f"{param_4[i][j]:.2f}", ha='right', va='top', fontsize=11)
            ax.text(j+0.2, i+0.8, f"{param_4[i][j]:.2f}", ha='right', va='top', fontsize=11)
            ax.text(j+0.2, i+0.2, f"{param_5[i][j]:.2f}", ha='right', va='top', fontsize=11)
    # plt.title(f'{title} - Param 1 & Param 2')
    
    mytitle = 'NICE:priv:{}/distance:{}/plaus_distance:{}/correctness:{}'.format(nice_param[index_1],nice_param[index_2],nice_param[index_3],nice_param[index_4])
    plt.title(mytitle)
    plt.xlabel('Elsilon')
    plt.ylabel('k')
    plt.xticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons)
    plt.yticks(ticks=np.arange(len(ks)) + 0.5, labels=ks, rotation=0)
    
    plt.savefig(plot_file)
    plt.close()

        ########################################################################
    ############ Draw improvement heatmaps ############################
    ########################################################################
    imp_1 =  np.zeros((len(ks),len(epsilons)))
    imp_2 =  np.zeros((len(ks),len(epsilons)))
    imp_3 =  np.zeros((len(ks),len(epsilons)))
    imp_4 =  np.zeros((len(ks),len(epsilons)))
    imp_5 =  np.zeros((len(ks),len(epsilons)))
    for i in range(len(ks)):
        for j in range(len(epsilons)):
            if index_1 in higher_better:
                imp_1[i][j] = ((param_1[i][j] - nice_param[index_1]) / ((nice_param[index_1])+0.000000000001))*100
            else:
                imp_1[i][j] = ((nice_param[index_1] - param_1[i][j]) / ((nice_param[index_1])+0.000000000001))*100
            if index_2 in higher_better:
                imp_2[i][j] = ((param_2[i][j] - nice_param[index_2]) / ((nice_param[index_2])+0.000000000001))*100
            else:
                imp_2[i][j] = ((nice_param[index_2] - param_2[i][j]) / ((nice_param[index_2])+0.000000000001))*100
            if index_3 in higher_better:
                imp_3[i][j] = ((param_3[i][j] - nice_param[index_3]) / ((nice_param[index_3])+0.000000000001))*100
            else:
                imp_3[i][j] = ((nice_param[index_3] - param_3[i][j]) / ((nice_param[index_3])+0.000000000001))*100
            if index_4 in higher_better:
                imp_4[i][j] = ((param_4[i][j] - nice_param[index_4]) / ((nice_param[index_4])+0.000000000001))*100
            else:
                imp_4[i][j] = ((nice_param[index_4] - param_4[i][j]) / ((nice_param[index_4])+0.000000000001))*100
            if index_5 in higher_better:
                imp_5[i][j] = ((param_5[i][j] - nice_param[index_5]) / ((nice_param[index_5])+0.000000000001))*100
            else:
                imp_5[i][j] = ((nice_param[index_5] - param_5[i][j]) / ((nice_param[index_5])+0.000000000001))*100


    plt.figure(figsize=(16,7))
    cbar_title = {'label': f'{title}  improvement'} 
    norm = plt.Normalize(vmin=imp_1.min(), vmax=imp_1.max())
    purples_palette = sns.light_palette("purple", as_cmap=True)
    blue_palette = sns.light_palette("blue", as_cmap=True)
    ax = sns.heatmap(imp_1,  cmap=purples_palette,  fmt='', norm = norm,cbar_kws=cbar_title)
    
    def get_text_color(background_color):
        r, g, b, _ = background_color
        luminance = 0.299*r + 0.587*g + 0.114*b
        return 'white' if luminance < 0.5 else 'black'

    for i in range(len(ks)):
        for j in range(len(epsilons)):
            background_color = purples_palette(norm(imp_1[i][j]))
            text_color = get_text_color(background_color)
            # Main value (param_1), larger and centered
            ax.text(j+.5, i+.5, f"{imp_1[i][j]:.2f}", ha='center', va='center', fontsize=11 ,color=text_color)
            # Secondary value (param_2), smaller and in the corners
            ax.text(j+0.8, i+0.2, f"{imp_2[i][j]:.1f}", ha='right', va='top', fontsize=11,color=text_color)
            ax.text(j+0.8, i+0.8, f"{imp_3[i][j]:.1f}", ha='right', va='top', fontsize=11,color=text_color)
            ax.text(j+0.3, i+0.8, f"{imp_4[i][j]:.1f}", ha='right', va='top', fontsize=11,color=text_color)
            ax.text(j+0.3, i+0.2, f"{imp_5[i][j]:.1f}", ha='right', va='top', fontsize=11,color=text_color)



    
    # plt.title={'label': f'{title} improvement'}
    plt.xlabel('Epsilon values')
    plt.ylabel('K values')
    plt.xticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons)
    plt.yticks(ticks=np.arange(len(ks)) + 0.5, labels=ks, rotation=0)
    plt.savefig(imp_plot_file)
    plt.close()


def plot_utility_privacy_tradeoff(CF_DF,graphs_dir,RANDOM_SEED,epsilons = None,ks = None ,method_name='Exp_prox_DP'):
    # plot utility/privacy tradeoff
        # record['average_distance'], 0
        # record['average_reidentification_rate'], 1
        # record['not_matching_count'],  2
        # record['one_exact_match'],  3
        # record['average_CF_min_dist'],  4
        # record['average_CF_min_k_dist'],  5
        # record['average_CF_rand_k_dist'],  6
        # record['success_rate'],  7
        # record['Have_match_count']  8
        # record['std_distance'], 9
        # record['std_CF_min_k_dist'] 10
    
    if epsilons is None and ks is None and method_name == 'NICE':
        # It is nice
        {
            # no graph, just show 
        }
    elif epsilons is not None and ks is None:
        # it is post_proc LDP or DP_DS
        # 1 dimentional heatmap for all epsilons all combinations of privacy/utility measuers
        # 1 . distance Vs reidentification_rate
        draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 0, index_1 = 1 ,title='reidentification_rate Vs distance',higher_better=False)
        # 2. distance Vs not_matching_count
        draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 0, index_1 = 2 ,title='not_matching_count  Vs distance',higher_better=True)
        # 3. distance Vs one_exact_match
        draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 0, index_1 = 3 ,title='one_exact_match  Vs distance',higher_better=False)
        # 4. distance Vs Have_match_count
        draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 0, index_1 = 8 ,title='Have_match_count  Vs distance',higher_better=False)
        # 5. success_rate Vs reidentification_rate
        draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 7, index_1 = 1 ,title='reidentification_rate Vs success_rate',higher_better=False)
        # 6. success_rate Vs not_matching_count
        draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 7, index_1 = 2 ,title='not_matching_count Vs success_rate',higher_better=True)
        # 7. success_rate Vs one_exact_match
        draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 7, index_1 = 3 ,title='one_exact_match Vs success_rate',higher_better=False)
        # 8. success_rate Vs Have_match_count
        draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 7, index_1 = 8 ,title='Have_match_count Vs success_rate',higher_better=False)
        # 9. average_CF_min_dist Vs reidentification_rate
        draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 4, index_1 = 1 ,title='reidentification_rate Vs average_CF_min_dist',higher_better=False)
        # 10. average_CF_min_dist Vs not_matching_count
        draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 4, index_1 = 2 ,title='not_matching_count Vs average_CF_min_dist',higher_better=True)
        # 11. average_CF_min_dist Vs one_exact_match
        draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 4, index_1 = 3 ,title='one_exact_match Vs average_CF_min_dist',higher_better=False)
        # 12. average_CF_min_dist Vs Have_match_count
        draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 4, index_1 = 8 ,title='Have_match_count Vs average_CF_min_dist',higher_better=False)
        # 13. average_CF_min_k_dist Vs reidentification_rate
        draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 5, index_1 = 1 ,title='reidentification_rate Vs average_CF_min_k_dist',higher_better=False)
        # 14. average_CF_min_k_dist Vs not_matching_count
        draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 5, index_1 = 2 ,title='not_matching_count Vs average_CF_min_k_dist',higher_better=True)
        # 15. average_CF_min_k_dist Vs one_exact_match
        draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 5, index_1 = 3 ,title='one_exact_match Vs average_CF_min_k_dist',higher_better=False)
        # 16. average_CF_min_k_dist Vs Have_match_count
        draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 5, index_1 = 8 ,title='Have_match_count Vs average_CF_min_k_dist',higher_better=False)
        # 17. average_CF_rand_k_dist Vs reidentification_rate
        draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 6, index_1 = 1 ,title='reidentification_rate Vs average_CF_rand_k_dist',higher_better=False)
        # 18. average_CF_rand_k_dist Vs not_matching_count
        draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 6, index_1 = 2 ,title='not_matching_count Vs average_CF_rand_k_dist',higher_better=True)
        # 19. average_CF_rand_k_dist Vs one_exact_match
        draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 6, index_1 = 3 ,title='one_exact_match Vs average_CF_rand_k_dist',higher_better=False)
        # 20. average_CF_rand_k_dist Vs Have_match_count
        draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 6, index_1 = 8 ,title='Have_match_count Vs average_CF_rand_k_dist',higher_better=False)
            
            
         
    elif epsilons is not None and ks is not None:
            # it is inline LDP or Exp_mech
            # it is post_proc LDP or DP_DS
            # 1 . distance Vs reidentification_rate
        draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 0, index_1 = 1 ,title='reidentification_rate Vs distance',higher_better=False)
        # 2. distance Vs not_matching_count
        draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 0, index_1 = 2 ,title='not_matching_count  Vs distance',higher_better=True)
        # 3. distance Vs one_exact_match
        draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 0, index_1 = 3 ,title='one_exact_match  Vs distance',higher_better=False)
        # 4. distance Vs Have_match_count
        draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 0, index_1 = 8 ,title='Have_match_count  Vs distance',higher_better=False)
        # 5. success_rate Vs reidentification_rate
        draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 7, index_1 = 1 ,title='reidentification_rate Vs success_rate',higher_better=False)
        # 6. success_rate Vs not_matching_count
        draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 7, index_1 = 2 ,title='not_matching_count Vs success_rate',higher_better=True)
        # 7. success_rate Vs one_exact_match
        draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 7, index_1 = 3 ,title='one_exact_match Vs success_rate',higher_better=False)
        # 8. success_rate Vs Have_match_count
        draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 7, index_1 = 8 ,title='Have_match_count Vs success_rate',higher_better=False)
        # 9. average_CF_min_dist Vs reidentification_rate
        draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 4, index_1 = 1 ,title='reidentification_rate Vs average_CF_min_dist',higher_better=False)
        # 10. average_CF_min_dist Vs not_matching_count
        draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 4, index_1 = 2 ,title='not_matching_count Vs average_CF_min_dist',higher_better=True)
        # 11. average_CF_min_dist Vs one_exact_match
        draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 4, index_1 = 3 ,title='one_exact_match Vs average_CF_min_dist',higher_better=False)
        # 12. average_CF_min_dist Vs Have_match_count
        draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 4, index_1 = 8 ,title='Have_match_count Vs average_CF_min_dist',higher_better=False)
        # 13. average_CF_min_k_dist Vs reidentification_rate
        draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 5, index_1 = 1 ,title='Vs reidentification_rate Vs average_CF_min_k_dist',higher_better=False)
        # 14. average_CF_min_k_dist Vs not_matching_count
        draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 5, index_1 = 2 ,title='not_matching_count Vs average_CF_min_k_dist',higher_better=True)
        # 15. average_CF_min_k_dist Vs one_exact_match
        draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 5, index_1 = 3 ,title='one_exact_match Vs average_CF_min_k_dist',higher_better=False)
        # 16. average_CF_min_k_dist Vs Have_match_count
        draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, ks,index_2= 5, index_1 = 8 ,title='Have_match_count Vs average_CF_min_k_dist',higher_better=False)
        # 17. average_CF_rand_k_dist Vs reidentification_rate
        draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, ks,index_2= 6, index_1 = 1 ,title='reidentification_rate Vs average_CF_rand_k_dist',higher_better=False)
        # 18. average_CF_rand_k_dist Vs not_matching_count
        draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 6, index_1 = 2 ,title=' not_matching_count Vs average_CF_rand_k_dist',higher_better=True)
        # 19. average_CF_rand_k_dist Vs one_exact_match
        draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 6, index_1 = 3 ,title='one_exact_match Vs average_CF_rand_k_dist',higher_better=False)
        # 20. average_CF_rand_k_dist Vs Have_match_count
        draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 6, index_1 = 8 ,title='Have_match_count Vs average_CF_rand_k_dist',higher_better=False)
    
    else:
            # Wrong setting
        print("Wrong setting in Draw_heatmap, please check the parameters and try again.")
        

def plot_utility_privacy_tradeoff_3_in_1(CF_DF,graphs_dir,RANDOM_SEED,epsilons = None,ks = None ,nice_param = None,method_name='Exp_prox_DP',higher_better = [2,7]):
    # plot utility/privacy tradeoff
        # record['average_distance'], 0
        # record['average_reidentification_rate'], 1
        # record['not_matching_count'],  2
        # record['one_exact_match'],  3
        # record['average_CF_min_dist'],  4
        # record['average_CF_min_k_dist'],  5
        # record['average_CF_rand_k_dist'],  6
        # record['success_rate'],  7
        # record['Have_match_count']  8
    if epsilons is None and ks is None and method_name == 'NICE':
        # It is nice
        {
            # no graph, just show 
        }
    elif epsilons is not None and ks is None:
        # it is post_proc LDP or DP_DS
        # 1 dimentional heatmap for all epsilons all combinations of privacy/utility measuers
        # 1 . distance Vs reidentification_rate
        # draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 0, index_1 = 1 ,title='reidentification_rate Vs distance',higher_better=False)
        ###########################3 params in one plot############################
        # 2. distance Vs not_matching_count
        # draw_one_dimensional_heatmap_3param(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,  index_1 = 2 ,index_2= 0,index_3 = 5,nice_param=nice_param, title='not_matching_count  Vs distance_plaus',higher_better=higher_better)
        # 3. distance Vs one_exact_match
        # draw_one_dimensional_heatmap_3param(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,  index_1 = 3 ,index_2= 0,index_3 = 5,nice_param=nice_param,title='one_exact_match  Vs distance_plaus',higher_better=higher_better)
        ###########################end params in one plot############################
        ###########################4 params in one plot############################
        # 2. distance Vs not_matching_count
        # draw_one_dimensional_heatmap_4param(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,  index_1 = 2 ,index_2= 0,index_3 = 5,index_4 = 7,nice_param=nice_param, title='M$_{0}$',higher_better=higher_better)
        # 3. distance Vs one_exact_match
        # draw_one_dimensional_heatmap_4param(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,  index_1 = 3 ,index_2= 0,index_3 = 5,index_4 =7,nice_param=nice_param,title= 'M$_{1}$',higher_better=higher_better)
        draw_one_dimensional_heatmap_5param(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,  index_1 = 2 ,index_2= 0,index_3 = 5,index_4 =7,index_5=3,nice_param=nice_param,title= 'M$_{0}$',higher_better=higher_better)
        draw_one_dimensional_heatmap_5param(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,  index_1 = 3 ,index_2= 0,index_3 = 5,index_4 =7,index_5=2,nice_param=nice_param,title= 'M$_{1}$',higher_better=higher_better)
        ###########################end params in one plot############################
        # 4. distance Vs Have_match_count
        # draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 0, index_1 = 8 ,title='Have_match_count  Vs distance',higher_better=False)
        # 5. success_rate Vs reidentification_rate
        # draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 7, index_1 = 1 ,title='reidentification_rate Vs success_rate',higher_better=False)
        # 6. success_rate Vs not_matching_count
        # draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 7, index_1 = 2 ,title='not_matching_count Vs success_rate',higher_better=True)
        # 7. success_rate Vs one_exact_match
        # draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 7, index_1 = 3 ,title='one_exact_match Vs success_rate',higher_better=False)
        # 8. success_rate Vs Have_match_count
        # draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 7, index_1 = 8 ,title='Have_match_count Vs success_rate',higher_better=False)
        # 9. average_CF_min_dist Vs reidentification_rate
        # draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 4, index_1 = 1 ,title='reidentification_rate Vs average_CF_min_dist',higher_better=False)
        # 10. average_CF_min_dist Vs not_matching_count
        # draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 4, index_1 = 2 ,title='not_matching_count Vs average_CF_min_dist',higher_better=True)
        # 11. average_CF_min_dist Vs one_exact_match
        # draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 4, index_1 = 3 ,title='one_exact_match Vs average_CF_min_dist',higher_better=False)
        # 12. average_CF_min_dist Vs Have_match_count
        # draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 4, index_1 = 8 ,title='Have_match_count Vs average_CF_min_dist',higher_better=False)
        # 13. average_CF_min_k_dist Vs reidentification_rate
        # draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 5, index_1 = 1 ,title='reidentification_rate Vs average_CF_min_k_dist',higher_better=False)
        # 14. average_CF_min_k_dist Vs not_matching_count
        # draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 5, index_1 = 2 ,title='not_matching_count Vs average_CF_min_k_dist',higher_better=True)
        # 15. average_CF_min_k_dist Vs one_exact_match
        # draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 5, index_1 = 3 ,title='one_exact_match Vs average_CF_min_k_dist',higher_better=False)
        # 16. average_CF_min_k_dist Vs Have_match_count
        # draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 5, index_1 = 8 ,title='Have_match_count Vs average_CF_min_k_dist',higher_better=False)
        # 17. average_CF_rand_k_dist Vs reidentification_rate
        # draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 6, index_1 = 1 ,title='reidentification_rate Vs average_CF_rand_k_dist',higher_better=False)
        # 18. average_CF_rand_k_dist Vs not_matching_count
        # draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 6, index_1 = 2 ,title='not_matching_count Vs average_CF_rand_k_dist',higher_better=True)
        # 19. average_CF_rand_k_dist Vs one_exact_match
        # draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 6, index_1 = 3 ,title='one_exact_match Vs average_CF_rand_k_dist',higher_better=False)
        # 20. average_CF_rand_k_dist Vs Have_match_count
        # draw_one_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_2= 6, index_1 = 8 ,title='Have_match_count Vs average_CF_rand_k_dist',higher_better=False)
            
            
         
    elif epsilons is not None and ks is not None:
            # it is inline LDP or Exp_mech
            # it is post_proc LDP or DP_DS
            # 1 . distance Vs reidentification_rate
        # draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 0, index_1 = 1 ,title='reidentification_rate Vs distance',higher_better=False)
        ###########################3 params in one plot############################
        # 2. distance Vs not_matching_count
        # draw_two_dimensional_heatmap_3param(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_1 = 2 ,index_2= 0,  index_3 = 5, nice_param=nice_param,title='not_matching_count  Vs distance_plaus',higher_better=higher_better)
        # 3. distance Vs one_exact_match
        # draw_two_dimensional_heatmap_3param(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_1 = 3, index_2= 0 , index_3 = 5 ,nice_param=nice_param,title='one_exact_match  Vs distance_plaus',higher_better=higher_better)
        ###########################end params in one plot############################
        ###########################4 params in one plot############################
        # 2. distance Vs not_matching_count
        # draw_two_dimensional_heatmap_4param(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_1 = 2 ,index_2= 0,  index_3 = 5,index_4 = 7, nice_param=nice_param,title='M$_{0}$',higher_better=higher_better)
        
        # 3. distance Vs one_exact_match
        draw_two_dimensional_heatmap_5param(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_1 = 3, index_2= 0 , index_3 = 5 ,index_4 = 7,index_5=2,nice_param=nice_param,title='M$_{1}$',higher_better=higher_better)
        draw_two_dimensional_heatmap_5param(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_1 = 2 ,index_2= 0,  index_3 = 5,index_4 = 7,index_5=3, nice_param=nice_param,title='M$_{0}$',higher_better=higher_better)
        ###########################end 4 params in one plot############################
        # 4. distance Vs Have_match_count
        # draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 0, index_1 = 8 ,title='Have_match_count  Vs distance',higher_better=False)
        # 5. success_rate Vs reidentification_rate
        # draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 7, index_1 = 1 ,title='reidentification_rate Vs success_rate',higher_better=False)
        # 6. success_rate Vs not_matching_count
        # draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 7, index_1 = 2 ,title='not_matching_count Vs success_rate',higher_better=True)
        # 7. success_rate Vs one_exact_match
        # draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 7, index_1 = 3 ,title='one_exact_match Vs success_rate',higher_better=False)
        # 8. success_rate Vs Have_match_count
        # draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 7, index_1 = 8 ,title='Have_match_count Vs success_rate',higher_better=False)
        # 9. average_CF_min_dist Vs reidentification_rate
        # draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 4, index_1 = 1 ,title='reidentification_rate Vs average_CF_min_dist',higher_better=False)
        # 10. average_CF_min_dist Vs not_matching_count
        # draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 4, index_1 = 2 ,title='not_matching_count Vs average_CF_min_dist',higher_better=True)
        # 11. average_CF_min_dist Vs one_exact_match
        # draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 4, index_1 = 3 ,title='one_exact_match Vs average_CF_min_dist',higher_better=False)
        # 12. average_CF_min_dist Vs Have_match_count
        # draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 4, index_1 = 8 ,title='Have_match_count Vs average_CF_min_dist',higher_better=False)
        # 13. average_CF_min_k_dist Vs reidentification_rate
        # draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 5, index_1 = 1 ,title='Vs reidentification_rate Vs average_CF_min_k_dist',higher_better=False)
        # 14. average_CF_min_k_dist Vs not_matching_count
        # draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 5, index_1 = 2 ,title='not_matching_count Vs average_CF_min_k_dist',higher_better=True)
        # 15. average_CF_min_k_dist Vs one_exact_match
        # draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 5, index_1 = 3 ,title='one_exact_match Vs average_CF_min_k_dist',higher_better=False)
        # 16. average_CF_min_k_dist Vs Have_match_count
        # draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, ks,index_2= 5, index_1 = 8 ,title='Have_match_count Vs average_CF_min_k_dist',higher_better=False)
        # 17. average_CF_rand_k_dist Vs reidentification_rate
        # draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, ks,index_2= 6, index_1 = 1 ,title='reidentification_rate Vs average_CF_rand_k_dist',higher_better=False)
        # 18. average_CF_rand_k_dist Vs not_matching_count
        # draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 6, index_1 = 2 ,title=' not_matching_count Vs average_CF_rand_k_dist',higher_better=True)
        # 19. average_CF_rand_k_dist Vs one_exact_match
        # draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 6, index_1 = 3 ,title='one_exact_match Vs average_CF_rand_k_dist',higher_better=False)
        # 20. average_CF_rand_k_dist Vs Have_match_count
        # draw_two_dimensional_heatmap(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, index_2= 6, index_1 = 8 ,title='Have_match_count Vs average_CF_rand_k_dist',higher_better=False)
    
    else:
            # Wrong setting
        print("Wrong setting in Draw_table, please check the parameters and try again.")

latex_representations = {
    "NICE": "$\\Nice$",
    "Exp_prox_DP": "$\dpnicem$", # "\texttt{Inline\_DP}", # "$Exp\_prox\_DP$",
    "Inline_DP":  "$\inlinedp$", #"$Inline_DP$",
    "DP_DS": "$\LDP$", # "$DP\_DS$",
    "Synth_DP_DS": "$\MST$",
    "Noisy_Max_DP": "$\\NoisyMax$",
    "DP_EXP_Feature": "$\expfeat$", #$DP\_EXP\_Feature$",
    "Laplace_Noise_DP": "$Laplace\_Noise\_DP$" #"$\LDPRR$" # "$Laplace\_Noise\_DP$"

    # Add more mappings as needed
}

def bold_max(s, is_max=True):
        """ Bold the maximum or minimum value in a column """
        if is_max:
            max_value = s.max()  # Bold the max value (for metrics where higher is better)
        else:
            max_value = s.min()  # Bold the min value (for metrics where lower is better)
        
        return [f"\\textbf{{{v:.2f}}}" if v == max_value else f"{v:.2f}" for v in s]
 


def draw_one_dimensional_table(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,nice_param):
        # record['average_distance'], 0
        # record['average_reidentification_rate'], 1
        # record['not_matching_count'],  2
        # record['one_exact_match'],  3
        # record['average_CF_min_dist'],  4
        # record['average_CF_min_k_dist'],  5
        # record['average_CF_rand_k_dist'],  6
        # record['success_rate'],  7
        # record['Have_match_count']  8
        # record['std_distance'],  9
        # record['std_CF_k_min_dist'], 10 

    # generate plot file path
    if RANDOM_SEED != 9999:  ### aggregated results
        plot_file_path = '{}{}/{}/'.format(graphs_dir,RANDOM_SEED,method_name)
    else:
        plot_file_path = '{}{}/'.format(graphs_dir,method_name)
    if not os.path.exists(plot_file_path):
        os.makedirs(plot_file_path, exist_ok=True)
    result_file = '{}/result.tex'.format(plot_file_path)
    imp_result_file = '{}/result_improvement.tex'.format(plot_file_path)

    print(CF_DF)
    temp_df = [[0]*9 for _ in range(len(epsilons)+1)]
    # temp_df = np.zeros((len(epsilons)+1,7))
    # extract data from DF_CF in order to create table
    nice_temp = np.zeros(10)
    temp_df[0][0] = "NICE"
    temp_df[0][1] = 0
    temp_df[0][2] = nice_param[0]
    temp_df[0][3] = nice_param[9]
    temp_df[0][4] = nice_param[5]
    temp_df[0][5] = nice_param[10]
    temp_df[0][6] = nice_param[7]
    temp_df[0][7] = nice_param[2]
    temp_df[0][8] = nice_param[3]
    
    
    for i in range(len(epsilons)):
        temp_df[i+1][0] =  method_name 
        temp_df[i+1][1] = epsilons[i]
        temp_df[i+1][2] = CF_DF[i][0] # average_distance
        temp_df[i+1][3] = CF_DF[i][9] # std_distance
        temp_df[i+1][4] = CF_DF[i][5] # average_CF_min_k_dist
        temp_df[i+1][5] = CF_DF[i][10] # std_CF_k_min_dist
        temp_df[i+1][6] = CF_DF[i][7] # Correctness
        temp_df[i+1][7] = CF_DF[i][2] # not_matching_count
        temp_df[i+1][8] = CF_DF[i][3] # one_exact_match
        
        
    
    
    
    epsilons_with_header = np.insert(epsilons, 0, 0)

    result_df = pd.DataFrame(temp_df,
                          index=epsilons_with_header, # pd.date_range(start="2021-01-01", periods=10),
                          columns=["Method","Epsilon", "$Dist_{prox} \\downarrow$","$Var_{Dist}$","$Plaus_{prox} \\downarrow$","$Var_{plaus}$","Correctness $\\uparrow$","$M_0 \\uparrow$","$M_1 \\downarrow$"])
                                    
    # Draw table


# Apply the function to the DataFrame to format the columns
# Assume upward arrows mean "higher is better" and downward arrows mean "lower is better"
    result_df["$Dist_{prox} \\downarrow$"] = bold_max(result_df["$Dist_{prox} \\downarrow$"], is_max=False)
    result_df["$Plaus_{prox} \\downarrow$"] = bold_max(result_df["$Plaus_{prox} \\downarrow$"], is_max=False)
    result_df["Correctness $\\uparrow$"] = bold_max(result_df["Correctness $\\uparrow$"], is_max=True)
    result_df["$M_0 \\uparrow$"] = bold_max(result_df["$M_0 \\uparrow$"], is_max=True)
    result_df["$M_1 \\downarrow$"] = bold_max(result_df["$M_1 \\downarrow$"], is_max=False)


    result_df['Method'] = result_df['Method'].apply(lambda x: latex_representations.get(x, x))
    result_df['Method'] = result_df['Method'].mask(result_df['Method'].duplicated(), '')
    result_df['Epsilon'] = result_df['Epsilon'].mask(result_df['Epsilon'].duplicated(), '')
    # Draw table
    # latex_table = result_df.to_latex(index=False, escape=False,float_format="{:.2f}".format)
    # latex_table_hline = add_hlines(latex_table,method_name)

    # latex_table_hline = add_hlines(latex_table,method_name)
    # with open(result_file, 'w') as f:
    #     f.write(latex_table_hline)
        
    
    latex_table = result_df.to_latex(index=False, escape=False,float_format="{:.2f}".format)
    wrapped_latex_table = (
    "\\begin{table}[ht]\n"
    "\\centering\n"
    + latex_table +
    "\\caption{Your caption here.}\n"
    "\\end{table}\n"
    )
    with open(result_file, 'w') as f:
        f.write(wrapped_latex_table)
    

    print("result Table saved at: ",result_file)

                  
    # Create improvement table
    nice_temp = np.zeros(8)
    nice_temp[0] = 0
    nice_temp[1] = nice_param[0]
    nice_temp[2] = nice_param[9]
    nice_temp[3] = nice_param[5]
    nice_temp[4] = nice_param[10]
    
    nice_temp[5] = nice_param[7]
    nice_temp[6] = nice_param[2]
    nice_temp[7] = nice_param[3]
    higher_better = [4,6]
    imp_df = [[0]*8 for _ in range(len(epsilons))] #  np.zeros((len(epsilons),6))
    for i in range(len(epsilons)):
        imp_df[i][0] = epsilons[i]
        for j in range(1,8):
            if j in higher_better:
                imp_df[i][j] = ((temp_df[i+1][j+1] - nice_temp[j]) / ((nice_temp[j])+0.000000000001))*100
            else:
                imp_df[i][j] = ((nice_temp[j] - temp_df[i+1][j+1]) / ((nice_temp[j])+0.000000000001))*100

    imp_result_df = pd.DataFrame(imp_df,
                          index=epsilons, 
                          columns=["Epsilon", "$Dist_{prox} \\downarrow$","$Var_{Dist}$","$Plaus_{prox} \\downarrow$","$Var_{plaus}$","Correctness $\\uparrow$","$M_0 \\uparrow$","$M_1 \\downarrow$"])

    # Find bests
    imp_result_df["$Dist_{prox} \\downarrow$"] = bold_max(imp_result_df["$Dist_{prox} \\downarrow$"], is_max=True)
    imp_result_df["$Plaus_{prox} \\downarrow$"] = bold_max(imp_result_df["$Plaus_{prox} \\downarrow$"], is_max=True)
    imp_result_df["Correctness $\\uparrow$"] = bold_max(imp_result_df["Correctness $\\uparrow$"], is_max=True)
    imp_result_df["$M_0 \\uparrow$"] = bold_max(imp_result_df["$M_0 \\uparrow$"], is_max=True)
    imp_result_df["$M_1 \\downarrow$"] = bold_max(imp_result_df["$M_1 \\downarrow$"], is_max=True)
    
    
    
    
    imp_result_df['Epsilon'] = imp_result_df['Epsilon'].mask(imp_result_df['Epsilon'].duplicated(), '')


    imp_latex_table = imp_result_df.to_latex(index=False, escape=False,float_format="{:.2f}".format)
    imp_latex_table_hline = add_hlines(imp_latex_table,method_name)
    #with open(imp_result_file, 'w') as f:
    #    f.write(imp_latex_table_hline)
     
    # latex_table = result_df.to_latex(index=False, escape=False,float_format="{:.2f}".format)
    wrapped_latex_table = (
    "\\begin{table}[ht]\n"
    "\\centering\n"
    + imp_latex_table_hline +
    "\\caption{Your caption here.}\n"
    "\\end{table}\n"
    )
    imp_result_file = '{}/result_improvement.tex'.format(plot_file_path)
    with open(imp_result_file, 'w') as f:
        f.write(wrapped_latex_table)
    
    
    print("improvement Table saved at: ",imp_result_file)
    

    

    
def add_lines_between_methods(df,method_name,columns):
    
    latex_table = df.to_latex(index=False, escape=False )
    lines = latex_table.split('\n')
    new_lines = []
    previous_method = None
    for line in lines:
        if previous_method is not None and line.startswith('NICE') and previous_method != 'NICE':
            new_lines.append('\\hline')
        if line.startswith('NICE') or line.startswith(method_name):
            previous_method = line.split('&')[0].strip()
    return '\n'.join(new_lines)

# def add_lines_between_methods(latex_table,method_name,columns):
#     lines = latex_table.split('\n')
#     new_lines = []
#     previous_method = None
#     for line in lines:
#         if previous_method is not None and line.startswith('NICE') and previous_method != 'NICE':
#             new_lines.append('\\hline')
#         new_lines.append(line)
#         if line.startswith('NICE') or line.startswith(method_name):
#             previous_method = line.split('&')[0].strip()
#     return '\n'.join(new_lines)



# add lines to seperate different settings
def add_hlines(latex_table, method_name):
    m_name_string = latex_representations[method_name]
    lines = latex_table.split('\n')
    new_lines = []
    # previous_method = None
    previous_epsilon = None

    for line in lines:
        # Check for method name change
        if line.startswith(m_name_string):
            previous_epsilon = line.split('&')[1].strip()
            new_lines.append('\\hline')
            
            # previous_method = m_name_string

        # # Check for epsilon change, for the first line after NICE
        # if previous_epsilon is None and line.strip() and  line.split('&')[0].strip() == m_name_string:
        #     previous_epsilon = line.split('&')[1].strip()
        #     new_lines.append(line)
    
        # lines with epsilon change, add hline - not when it os NICE
        if previous_epsilon is not None and line.strip() and line.split('&')[0].strip()  == '' and line.split('&')[1].strip() != '': # not line.startswith('NICE'):
            current_epsilon = line.split('&')[1].strip()
            if current_epsilon != previous_epsilon:
                new_lines.append('\\hline')
                previous_epsilon = current_epsilon

        # new_lines.append(line)

        
        # if previous_epsilon is None and  line.strip() and line.split('&')[0].strip() in (m_name_string,''): #not line.startswith('NICE'):
        #     previous_epsilon = line.split('&')[1].strip()

        # add processed line to the updated latex table
        new_lines.append(line)
    return '\n'.join(new_lines)

##### Works well
# def add_hlines(latex_table,method_name):
#     m_name_string = '$'+method_name+'$'
#     lines = latex_table.split('\n')
#     new_lines = []
#     for line in lines:
#         if line.startswith(m_name_string): #line.strip() and not line.startswith('NICE') and line.startswith(method_name)#line.split('&')[0].strip():
#             new_lines.append('\\hline')
#         new_lines.append(line)
#     return '\n'.join(new_lines)

def draw_two_dimentional_table(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks,nice_param):
    if RANDOM_SEED != 9999:  ### aggregated results
        plot_file_path = '{}/{}/{}/'.format(graphs_dir,RANDOM_SEED,method_name)
    else:
        plot_file_path = '{}/{}/'.format(graphs_dir,method_name)

    if not os.path.exists(plot_file_path):
        os.makedirs(plot_file_path, exist_ok=True)
    result_file = '{}/results.tex'.format(plot_file_path)
    imp_result_file = '{}/result_improvement.tex'.format(plot_file_path)

    

    temp_df = [[0]*10 for _ in range((len(epsilons)*len(ks))+1)]
    # temp_df = np.zeros((len(epsilons)+1,7))
    # extract data from DF_CF in order to create table
    #first row is NICE
    nice_temp = np.zeros(8)
    temp_df[0][0] = "NICE"
    temp_df[0][1] = 0
    temp_df[0][2] = 1
    temp_df[0][3] = nice_param[0]
    temp_df[0][4] = nice_param[9]
    temp_df[0][5] = nice_param[5]
    temp_df[0][6] = nice_param[10]
    temp_df[0][7] = nice_param[7]
    temp_df[0][8] = nice_param[2]
    temp_df[0][9] = nice_param[3]
    
    for i in range(len(epsilons)):
        for j in range(len(ks)):
            temp_df[i*len(ks)+j+1][0] = method_name
            temp_df[i*len(ks)+j+1][1] = epsilons[i]
            temp_df[i*len(ks)+j+1][2] = ks[j]
            temp_df[i*len(ks)+j+1][3] = CF_DF[i][j][0] # average_distance
            temp_df[i*len(ks)+j+1][4] = CF_DF[i][j][9] # variance_distance
            temp_df[i*len(ks)+j+1][5] = CF_DF[i][j][5] # average_CF_min_k_dist
            temp_df[i*len(ks)+j+1][6] = CF_DF[i][j][10] # variance_CF_min_k _dist
            temp_df[i*len(ks)+j+1][7] = CF_DF[i][j][7] # Correctness
            temp_df[i*len(ks)+j+1][8] = CF_DF[i][j][2] # not_matching_count
            temp_df[i*len(ks)+j+1][9] = CF_DF[i][j][3] # one_exact_match
        # temp_df[i+1][0] = method_name
        # temp_df[i+1][1] = epsilons[i]
        # temp_df[i+1][2] = CF_DF[i][0] # average_distance
        # temp_df[i+1][3] = CF_DF[i][5] # average_CF_min_k_dist
        # temp_df[i+1][4] = CF_DF[i][7] # Correctness
        # temp_df[i+1][5] = CF_DF[i][2] # not_matching_count
        # temp_df[i+1][6] = CF_DF[i][3] # one_exact_match
    
    
    
    # epsilons_with_header = np.insert(epsilons, 0, 0)

    # columns = ["Method","Epsilon","K", "$Dist_{prox} \\downarrow$","$Plaus_{prox} \\downarrow$","Correctness $\\uparrow$","$M_0 \\uparrow$","$M_1 \\downarrow$"]
    columns=["Method","Epsilon","K", "$Dist_{prox} \\downarrow$","$Var_{Dist_}$","$Plaus_{prox} \\downarrow$","$Var_{Plaus}$","Correctness $\\uparrow$","$M_0 \\uparrow$","$M_1 \\downarrow$"]
    result_df = pd.DataFrame(temp_df, columns=columns)
    # result_df = pd.DataFrame(temp_df,
                        #   index=epsilons_with_header, # pd.date_range(start="2021-01-01", periods=10),
                        #   columns=["Method","epsilon","K", "$Dist_{prox} \\downarrow$","$plaus_{prox} \\downarrow$","Correctness $\\uparrow$","$M_0 \\uparrow$","$M_1 \\downarrow$"])

    result_df["$Dist_{prox} \\downarrow$"] = bold_max(result_df["$Dist_{prox} \\downarrow$"], is_max=False)
    result_df["$Plaus_{prox} \\downarrow$"] = bold_max(result_df["$Plaus_{prox} \\downarrow$"], is_max=False)
    result_df["Correctness $\\uparrow$"] = bold_max(result_df["Correctness $\\uparrow$"], is_max=True)
    result_df["$M_0 \\uparrow$"] = bold_max(result_df["$M_0 \\uparrow$"], is_max=True)
    result_df["$M_1 \\downarrow$"] = bold_max(result_df["$M_1 \\downarrow$"], is_max=False)
                                
    result_df['Method'] = result_df['Method'].apply(lambda x: latex_representations.get(x, x))
    result_df['Method'] = result_df['Method'].mask(result_df['Method'].duplicated(), '')
    result_df['Epsilon'] = result_df['Epsilon'].mask(result_df['Epsilon'].duplicated(), '')
    # Draw table
    latex_table = result_df.to_latex(index=False, escape=False,float_format="{:.2f}".format)
    latex_table_hline = add_hlines(latex_table,method_name)
    wrapped_latex_table = (
    "\\begin{table}[ht]\n"
    "\\centering\n"
    + latex_table_hline +
    "\\caption{Your caption here.}\n"
    "\\end{table}\n"
    )
    with open(result_file, 'w') as f:
        f.write(wrapped_latex_table)

    # print(latex_table_hline)

        
    # Create improvement table
    nice_temp = np.zeros(9)
    nice_temp[0] = 0 #epsilon
    nice_temp[1] = 1 # k
    nice_temp[2] = nice_param[0]
    nice_temp[3] = nice_param[9]
    nice_temp[4] = nice_param[5]
    nice_temp[6] = nice_param[10]
    nice_temp[6] = nice_param[7]
    nice_temp[7] = nice_param[2]
    nice_temp[8] = nice_param[3]
    higher_better = [6,7]
    imp_df = [[0]*9 for _ in range((len(epsilons)*len(ks)))]
    # imp_df = [[0]*6 for _ in range(len(epsilons))] #  np.zeros((len(epsilons),6))
    for i in range(len(epsilons)):
        for k in range(len(ks)):
            imp_df[i*len(ks)+k][0] = epsilons[i]
            imp_df[i*len(ks)+k][1] = ks[k]

            # imp_df[k+(i*k)][0] = epsilons[i]
            for j in range(2,9):    # iterate over the columns
                if j in higher_better:
                    imp_df[k+(i*len(ks))][j] = ((temp_df[k+(i*len(ks))+1][j+1] - nice_temp[j]) / ((nice_temp[j])+0.000000000001))*100
                else:
                    imp_df[k+(i*len(ks))][j] = ((nice_temp[j] - temp_df[k+(i*len(ks))+1][j+1]) / ((nice_temp[j])+0.000000000001))*100

    imp_result_df = pd.DataFrame(imp_df,
                          columns=["Epsilon","K", "$Dist_{prox} \\downarrow$","$Var_{Dist}$","$plaus_{prox} \\downarrow$","$Var_{plaus}$","Correctness $\\uparrow$","$M_0 \\uparrow$","$M_1 \\downarrow$"])
    # imp_result_df.set_index(['Epsilon', 'K'], inplace=True)
    # Find bests
    imp_result_df["$Dist_{prox} \\downarrow$"] = bold_max(imp_result_df["$Dist_{prox} \\downarrow$"], is_max=True)
    imp_result_df["$plaus_{prox} \\downarrow$"] = bold_max(imp_result_df["$plaus_{prox} \\downarrow$"], is_max=True)
    imp_result_df["Correctness $\\uparrow$"] = bold_max(imp_result_df["Correctness $\\uparrow$"], is_max=True)
    imp_result_df["$M_0 \\uparrow$"] = bold_max(imp_result_df["$M_0 \\uparrow$"], is_max=True)
    imp_result_df["$M_1 \\downarrow$"] = bold_max(imp_result_df["$M_1 \\downarrow$"], is_max=True)
    
    # imp_result_df['Method'] = imp_result_df['Method'].apply(lambda x: latex_representations.get(x, x))
    # imp_result_df['Method'] = imp_result_df['Method'].mask(imp_result_df['Method'].duplicated(), '')
    imp_result_df['Epsilon'] = imp_result_df['Epsilon'].mask(imp_result_df['Epsilon'].duplicated(), '')


    imp_latex_table = imp_result_df.to_latex(index=False, escape=False,float_format="{:.2f}".format)
    imp_latex_table_hline = add_hlines(imp_latex_table,method_name)
    
    
    wrapped_latex_table = (
    "\\begin{table}[ht]\n"
    "\\centering\n"
    + imp_latex_table_hline +
    "\\caption{Your caption here.}\n"
    "\\end{table}\n"
    )
    
    with open(imp_result_file, 'w') as f:
        f.write(wrapped_latex_table)
    

    print("Improvement Table saved at: ",imp_result_file)

    # imp_result_df.to_latex(plot_file_path+'result_improvement_table.tex',index=False, float_format="{:.2f}".format)
    # plt.close()
    
                                
    
    # Draw table
    # latex_table = result_df.to_latex(index=False, escape=False,float_format="{:.2f}".format)
    # latex_table_hline = add_hlines(latex_table,method_name)
    # with open(result_file, 'w') as f:
    #     f.write(latex_table_hline)
    # print("Improvement Table saved at: ",plot_file_path)
    
    
    
    
    
    
    
    
    ########################################################################
    # 3 param heatmaps                                              ########
    ########################################################################
def draw_one_dimensional_heatmap_3param(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_1, index_2 , index_3 ,nice_param,title,higher_better):
    # generate plot file path
    if RANDOM_SEED != 9999:  ### aggregated results
        plot_file_path = '{}/{}/{}/'.format(graphs_dir,RANDOM_SEED,method_name)
    else:
        plot_file_path = '{}/{}/'.format(graphs_dir,method_name)
    if not os.path.exists(plot_file_path):
        os.makedirs(plot_file_path, exist_ok=True)
    plot_file = '{}/{}.png'.format(plot_file_path,title)
    imp_plot_file = '{}/{}_improvement.png'.format(plot_file_path,title)

    # draw plot 
    # param_1 = CF_DF.iloc[:,index1]
    # param_2 = CF_DF.iloc[:,index_2]
    # plt.figure(figsize=(14, 6))
    # sns.heatmap(pd.DataFrame({'param_1': param_1, 'param_2': param_2}), annot=True, cmap="viridis", cbar_kws={'label': 'Param 1 Value'}, fmt='d')
    # plt.title(title)
    param_1 = np.zeros(len(epsilons))
    param_2 = np.zeros(len(epsilons))
    param_3 = np.zeros(len(epsilons))
    for i in range(len(epsilons)):
        param_1[i] = CF_DF[i][index_1]
        param_2[i]= CF_DF[i][index_2]
        param_3[i]= CF_DF[i][index_3]
    # Create a 2D array for heatmap where each row is the param value repeated
    # This is a workaround to use seaborn's heatmap for one-dimensional data
    # combined_annotations = np.array([[f"{param_1[i]:.2f}\n{param_2[i]:.2f}"] for i in range(len(epsilons))])
    
    plt.figure(figsize=(8, 3))  # Adjusted figure size for one-dimensional data
    if index_1 in higher_better:
        # sns.heatmap(param_1.reshape(-1,1), annot=combined_annotations, cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
        ax = sns.heatmap(param_1.reshape(1,-1), cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
    else:
        # sns.heatmap(param_1.reshape(-1,1), annot=combined_annotations, cmap="Blues",  fmt='', vmin=param_1.max(), vmax=param_1.min())
        ax = sns.heatmap(param_1.reshape(1,-1),  cmap="Blues",  fmt='', vmin=param_1.max(), vmax=param_1.min())

    for i in range(len(epsilons)):
    
        # Main value (param_1), larger and centered
        ax.text(i+.4,.5, f"{param_1[i]:.2f}", ha='center', va='center', fontsize=10)
        # Secondary value (param_2), smaller and in the top-right corner
        # This is a workaround; precise positioning like top-right is limited
        ax.text( i+0.8, .3,f"{param_2[i]:.2f}", ha='right', va='top', fontsize=6)
        ax.text( i+0.8, .7,f"{param_3[i]:.2f}", ha='right', va='bottom', fontsize=6)


    # sns.heatmap(param_1, annot=combined_annotations, cmap="viridis", cbar_kws={'label': 'Param 1 Value'}, fmt='')
    # plt.title(f'{title} - Param')
    mytitle = 'NICE:priv:{}/distance:{}/plaus_distance:{}/'.format(nice_param[index_1],nice_param[index_2],nice_param[index_3])
    plt.title(mytitle)
    plt.xlabel('Epsilon values')
    plt.xticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons)
    plt.savefig(plot_file)
    plt.close()
    
    ########################################################################
    ############ Draw improvement heatmaps ############################
    ########################################################################
    imp_1 = np.zeros(len(epsilons))
    imp_2 = np.zeros(len(epsilons))
    imp_3 = np.zeros(len(epsilons))
    for i in range(len(epsilons)):
        if index_1 in higher_better:
            imp_1[i] = (param_1[i] - nice_param[index_1]) / nice_param[index_1]
        else:
            imp_1[i] = (nice_param[index_1] - param_1[i]) / nice_param[index_1]
        if index_2 in higher_better:
            imp_2[i] = (param_2[i] - nice_param[index_2]) / nice_param[index_2]
        else:
            imp_2[i] = (nice_param[index_2] - param_2[i]) / nice_param[index_2]
        if index_3 in higher_better:
            imp_3[i] = (param_3[i] - nice_param[index_3]) / nice_param[index_3]
        else:
            imp_3[i] = (nice_param[index_3] - param_3[i]) / nice_param[index_3]


    plt.figure(figsize=(6, 2))  # Adjusted figure size for one-dimensional data
        # sns.heatmap(param_1.reshape(-1,1), annot=combined_annotations, cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
    ax = sns.heatmap(imp_1.reshape(1,-1), cmap="Purples",  fmt='', vmin=imp_1.min(), vmax=imp_1.max())
    
    for i in range(len(epsilons)):
    
        # Main value (param_1), larger and centered
        ax.text(i+.4,.5, f"{imp_1[i]:.2f}", ha='center', va='center', fontsize=10)
        # Secondary value (param_2), smaller and in the top-right corner
        # This is a workaround; precise positioning like top-right is limited
        ax.text( i+0.8, .3,f"{imp_2[i]:.2f}", ha='right', va='top', fontsize=6)
        ax.text( i+0.8, .7,f"{imp_3[i]:.2f}", ha='right', va='bottom', fontsize=6)


    # sns.heatmap(param_1, annot=combined_annotations, cmap="viridis", cbar_kws={'label': 'Param 1 Value'}, fmt='')
    # plt.title(f'{title} - Param')
    plt.xlabel('Epsilon values')
    plt.xticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons)
    
    plt.savefig(imp_plot_file)
    plt.close()

    

def draw_two_dimensional_heatmap_3param(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks ,index_1, index_2 , index_3 ,nice_param,title,higher_better):
    # generate plot file path
    if RANDOM_SEED != 9999:  ### aggregated results
        plot_file_path = '{}/{}/{}/'.format(graphs_dir,RANDOM_SEED,method_name)
    else:
        plot_file_path = '{}/{}/'.format(graphs_dir,method_name)

    if not os.path.exists(plot_file_path):
        os.makedirs(plot_file_path, exist_ok=True)
    plot_file = '{}/{}.png'.format(plot_file_path,title)
    imp_plot_file = '{}/{}_improvement.png'.format(plot_file_path,title)
    # CF_DF is 2 dimensional array so, param_1 and param_2 are 2 dimensional arrays
    # param_1 = np.zeros((len(epsilons), len(ks)))
    # param_2 = np.zeros((len(epsilons), len(ks)))
    # param_3 = np.zeros((len(epsilons), len(ks)))
    # for i in range(len(epsilons)):
    #     for j in range(len(ks)):
    #         param_1[i][j] = CF_DF[i][j][index_1]
    #         param_2[i][j] = CF_DF[i][j][index_2]
    #         param_3[i][j] = CF_DF[i][j][index_3]
    param_1 = np.zeros((len(ks),len(epsilons)))
    param_2 = np.zeros((len(ks),len(epsilons)))
    param_3 = np.zeros((len(ks),len(epsilons)))
    for i in range(len(ks)):
        for j in range(len(epsilons)):
            param_1[i][j] = CF_DF[j][i][index_1]
            param_2[i][j] = CF_DF[j][i][index_2]
            param_3[i][j] = CF_DF[j][i][index_3]
    # combined_annotations = np.array([[f"{param_1[i][j]:.2f}\n{param_2[i][j]:.2f}" for j in range(len(ks))] for i in range(len(epsilons))])

    plt.figure(figsize=(16, 7))
    if index_1 in higher_better:
        # sns.heatmap(param_1, annot=combined_annotations, cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
        ax = sns.heatmap(param_1,  cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
    else:
        # sns.heatmap(param_1, annot=combined_annotations, cmap="Blues",  fmt='', vmin=param_1.max(), vmax=param_1.min())
        ax = sns.heatmap(param_1, cmap="Blues",  fmt='', vmin=param_1.max(), vmax=param_1.min())
    
    # for i in range(len(epsilons)):
    #     for j in range(len(ks)):
    #         # Main value (param_1), larger and centered
    #         ax.text(j+.4, i+.5, f"{param_1[i][j]:.2f}", ha='center', va='center', fontsize=12)
    #         # Secondary value (param_2), smaller and in the top-right corner
    #         # This is a workaround; precise positioning like top-right is limited
    #         ax.text(j+0.8, i+0.3, f"{param_2[i][j]:.2f}", ha='right', va='top', fontsize=9)
    #         ax.text(j+0.8, i+0.7, f"{param_3[i][j]:.2f}", ha='right', va='top', fontsize=9)

    for i in range(len(ks)):
        for j in range(len(epsilons)):
            # Main value (param_1), larger and centered
            ax.text(j+.4, i+.5, f"{param_1[i][j]:.2f}", ha='center', va='center', fontsize=12)
            # Secondary value (param_2), smaller and in the top-right corner
            # This is a workaround; precise positioning like top-right is limited
            ax.text(j+0.8, i+0.3, f"{param_2[i][j]:.2f}", ha='right', va='top', fontsize=9)
            ax.text(j+0.8, i+0.7, f"{param_3[i][j]:.2f}", ha='right', va='top', fontsize=9)


    # plt.title(f'{title} - Param 1 & Param 2')
    mytitle = 'NICE:priv:{}/distance:{}/plaus_distance:{}/'.format(nice_param[index_1],nice_param[index_2],nice_param[index_3])
    plt.title(mytitle)
    plt.xlabel('Elsilon')
    plt.ylabel('k')
    plt.xticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons)
    plt.yticks(ticks=np.arange(len(ks)) + 0.5, labels=ks, rotation=0)
    
    plt.savefig(plot_file)
    plt.close()

        ########################################################################
    ############ Draw improvement heatmaps ############################
    ########################################################################
    imp_1 =  np.zeros((len(ks),len(epsilons)))
    imp_2 =  np.zeros((len(ks),len(epsilons)))
    imp_3 =  np.zeros((len(ks),len(epsilons)))
    for i in range(len(ks)):
        for j in range(len(epsilons)):
            if index_1 in higher_better:
                imp_1[i][j] = (param_1[i][j] - nice_param[index_1]) / nice_param[index_1]
            else:
                imp_1[i][j] = (nice_param[index_1] - param_1[i][j]) / nice_param[index_1]
            if index_2 in higher_better:
                imp_2[i][j] = (param_2[i][j] - nice_param[index_2]) / nice_param[index_2]
            else:
                imp_2[i][j] = (nice_param[index_2] - param_2[i][j]) / nice_param[index_2]
            if index_3 in higher_better:
                imp_3[i][j] = (param_3[i][j] - nice_param[index_3]) / nice_param[index_3]
            else:
                imp_3[i][j] = (nice_param[index_3] - param_3[i][j]) / nice_param[index_3]


    plt.figure(figsize=(14, 6))
       
    ax = sns.heatmap(param_1,  cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
    
    for i in range(len(ks)):
        for j in range(len(epsilons)):
            # Main value (param_1), larger and centered
            ax.text(j+.4, i+.5, f"{imp_1[i][j]}", ha='center', va='center', fontsize=12)
            # Secondary value (param_2), smaller and in the top-right corner
            # This is a workaround; precise positioning like top-right is limited
            ax.text(j+0.8, i+0.3, f"{imp_2[i][j]}", ha='right', va='top', fontsize=9)
            ax.text(j+0.8, i+0.7, f"{imp_3[i][j]}", ha='right', va='top', fontsize=9)



    # sns.heatmap(param_1, annot=combined_annotations, cmap="viridis", cbar_kws={'label': 'Param 1 Value'}, fmt='')
    # plt.title(f'{title} - Param')
    plt.xlabel('Epsilon values')
    plt.xticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons)
    
    plt.savefig(imp_plot_file)
    plt.close()


def draw_one_dimensional_heatmap_4param(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_1, index_2 , index_3 ,index_4,nice_param,title,higher_better):
    # generate plot file path
    if RANDOM_SEED != 9999:  ### aggregated results
        plot_file_path = '{}/{}/{}/'.format(graphs_dir,RANDOM_SEED,method_name)
    else:
        plot_file_path = '{}/{}/'.format(graphs_dir,method_name)
    if not os.path.exists(plot_file_path):
        os.makedirs(plot_file_path, exist_ok=True)
    plot_file = '{}/{}.png'.format(plot_file_path,title)
    imp_plot_file = '{}/{}_improvement.png'.format(plot_file_path,title)

    
    param_1 = np.zeros(len(epsilons))
    param_2 = np.zeros(len(epsilons))
    param_3 = np.zeros(len(epsilons))
    param_4 = np.zeros(len(epsilons))
    for i in range(len(epsilons)):
        param_1[i] = CF_DF[i][index_1]
        param_2[i]= CF_DF[i][index_2]
        param_3[i]= CF_DF[i][index_3]
        param_4[i]= CF_DF[i][index_4]
    # Create a 2D array for heatmap where each row is the param value repeated
    # This is a workaround to use seaborn's heatmap for one-dimensional data
    # combined_annotations = np.array([[f"{param_1[i]:.2f}\n{param_2[i]:.2f}"] for i in range(len(epsilons))])
    
    plt.figure(figsize=(8, 2))  # Adjusted figure size for one-dimensional data
    cbar_title = {'label': f'{title}'}
    purples_palette = sns.light_palette("purple", as_cmap=True)
    blues_palette = sns.light_palette("blue", as_cmap=True)
    if index_1 in higher_better:
        # sns.heatmap(param_1.reshape(-1,1), annot=combined_annotations, cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
        ax = sns.heatmap(param_1.reshape(1,-1), cmap=purples_palette,  fmt='', vmin=param_1.min(), vmax=param_1.max(),cbar_kws=cbar_title)
    else:
        # sns.heatmap(param_1.reshape(-1,1), annot=combined_annotations, cmap="Blues",  fmt='', vmin=param_1.max(), vmax=param_1.min())
        ax = sns.heatmap(param_1.reshape(1,-1),  cmap=blues_palette,  fmt='', vmin=param_1.max(), vmax=param_1.min(),cbar_kws=cbar_title)

    for i in range(len(epsilons)):
    
        # Main value (param_1), larger and centered
        ax.text(i+.3,.5, f"{param_1[i]:.2f}", ha='center', va='center', fontsize=11)
        # Secondary value (param_2), smaller and in the top-right corner
        # This is a workaround; precise positioning like top-right is limited
        ax.text( i+0.8, .3,f"{param_2[i]:.2f}", ha='right', va='top', fontsize=7)
        ax.text( i+0.8, .7,f"{param_3[i]:.2f}", ha='right', va='top', fontsize=7)
        ax.text( i+0.2, .7,f"{param_4[i]:.2f}", ha='right', va='top', fontsize=7)


    # plt.title(f'{title} - Param')
    mytitle = 'NICE:priv:{}/distance:{}/plaus_distance:{}/correctness:{}'.format(nice_param[index_1],nice_param[index_2],nice_param[index_3],nice_param[index_4])
    plt.title(mytitle)
    plt.xlabel('Epsilon values')
    plt.xticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons)
    plt.savefig(plot_file)
    plt.close()
    
    ########################################################################
    ############ Draw improvement heatmaps ############################
    ########################################################################
    imp_1 = np.zeros(len(epsilons))
    imp_2 = np.zeros(len(epsilons))
    imp_3 = np.zeros(len(epsilons))
    imp_4 = np.zeros(len(epsilons))
    for i in range(len(epsilons)):
        if index_1 in higher_better:
            imp_1[i] = ((param_1[i] - nice_param[index_1]) / nice_param[index_1])*100
        else:
            imp_1[i] = ((nice_param[index_1] - param_1[i]) / nice_param[index_1])*100
        if index_2 in higher_better:
            imp_2[i] = ((param_2[i] - nice_param[index_2]) / nice_param[index_2])*100
        else:
            imp_2[i] = ((nice_param[index_2] - param_2[i]) / nice_param[index_2])*100
        if index_3 in higher_better:
            imp_3[i] = ((param_3[i] - nice_param[index_3]) / nice_param[index_3])*100
        else:
            imp_3[i] = ((nice_param[index_3] - param_3[i]) / nice_param[index_3])*100
        if index_4 in higher_better:
            imp_4[i] = ((param_4[i] - nice_param[index_4]) / nice_param[index_4])*100
        else:
            imp_4[i] = ((nice_param[index_4] - param_4[i]) / nice_param[index_4])*100


    plt.figure(figsize=(7, 2))  # Adjusted figure size for one-dimensional data
        # sns.heatmap(param_1.reshape(-1,1), annot=combined_annotations, cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
    cbar_title = {'label': f'{title}  improvement'} 
    norm = plt.Normalize(vmin=imp_1.min(), vmax=imp_1.max())
    purples_palette = sns.light_palette("purple", as_cmap=True)
    blues_palette = sns.light_palette("blue", as_cmap=True)
    ax = sns.heatmap(imp_1.reshape(1,-1), cmap=purples_palette,  fmt='', norm = norm,cbar_kws=cbar_title)
    
    for i in range(len(epsilons)):
    
        # Main value (param_1), larger and centered
        ax.text(i+.5,.5, f"{imp_1[i]:.2f}", ha='center', va='center', fontsize=10)
        # Secondary value (param_2), smaller and in the top-right corner
        ax.text( i+0.8, .3,f"{imp_2[i]:.1f}", ha='right', va='top', fontsize=7)
        ax.text( i+0.8, .7,f"{imp_3[i]:.1f}", ha='right', va='top', fontsize=7)
        ax.text( i+0.3, .7,f"{imp_4[i]:.1f}", ha='right', va='top', fontsize=7)


    # sns.heatmap(param_1, annot=combined_annotations, cmap="viridis", cbar_kws={'label': 'Param 1 Value'}, fmt='')
    # plt.title(f'{title} - Param')
    plt.cbar_kws={'label': f'{title} improvement'}
    plt.xlabel('Epsilon values')
    plt.xticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons)
    
    plt.savefig(imp_plot_file)
    plt.close()

    
def draw_two_dimensional_heatmap_4param(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks ,index_1, index_2 , index_3 ,index_4,nice_param,title,higher_better):
    # generate plot file path
    if RANDOM_SEED != 9999:  ### aggregated results
        plot_file_path = '{}/{}/{}/'.format(graphs_dir,RANDOM_SEED,method_name)
    else:
        plot_file_path = '{}/{}/'.format(graphs_dir,method_name)

    if not os.path.exists(plot_file_path):
        os.makedirs(plot_file_path, exist_ok=True)
    plot_file = '{}/{}.png'.format(plot_file_path,title)
    imp_plot_file = '{}/{}_improvement.png'.format(plot_file_path,title)
    # CF_DF is 2 dimensional array so, param_1 and param_2 are 2 dimensional arrays
    # param_1 = np.zeros((len(epsilons), len(ks)))
    # param_2 = np.zeros((len(epsilons), len(ks)))
    # param_3 = np.zeros((len(epsilons), len(ks)))
    # for i in range(len(epsilons)):
    #     for j in range(len(ks)):
    #         param_1[i][j] = CF_DF[i][j][index_1]
    #         param_2[i][j] = CF_DF[i][j][index_2]
    #         param_3[i][j] = CF_DF[i][j][index_3]
    param_1 = np.zeros((len(ks),len(epsilons)))
    param_2 = np.zeros((len(ks),len(epsilons)))
    param_3 = np.zeros((len(ks),len(epsilons)))
    param_4 = np.zeros((len(ks),len(epsilons)))
    for i in range(len(ks)):
        for j in range(len(epsilons)):
            param_1[i][j] = CF_DF[j][i][index_1]
            param_2[i][j] = CF_DF[j][i][index_2]
            param_3[i][j] = CF_DF[j][i][index_3]
            param_4[i][j] = CF_DF[j][i][index_4]
    # combined_annotations = np.array([[f"{param_1[i][j]:.2f}\n{param_2[i][j]:.2f}" for j in range(len(ks))] for i in range(len(epsilons))])

    plt.figure(figsize=(16, 7))
    cbar_title = {'label': f'{title}'}
    purples_palette = sns.light_palette("purple", as_cmap=True)
    blue_palette = sns.light_palette("blue", as_cmap=True)
    if index_1 in higher_better:
        # sns.heatmap(param_1, annot=combined_annotations, cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
        ax = sns.heatmap(param_1,  cmap=purples_palette,  fmt='', vmin=param_1.min(), vmax=param_1.max(),cbar_kws=cbar_title)
    else:
        # sns.heatmap(param_1, annot=combined_annotations, cmap="Blues",  fmt='', vmin=param_1.max(), vmax=param_1.min())
        ax = sns.heatmap(param_1, cmap=blue_palette,  fmt='', vmin=param_1.max(), vmax=param_1.min(),cbar_kws=cbar_title)
    
    
    for i in range(len(ks)):
        for j in range(len(epsilons)):
            # Main value (param_1), larger and centered
            ax.text(j+.5, i+.5, f"{param_1[i][j]:.2f}", ha='center', va='center', fontsize=12)
            # Secondary value (param_2), smaller and in the top-right corner
            # This is a workaround; precise positioning like top-right is limited
            ax.text(j+0.8, i+0.3, f"{param_2[i][j]:.2f}", ha='right', va='top', fontsize=9)
            ax.text(j+0.8, i+0.7, f"{param_3[i][j]:.2f}", ha='right', va='top', fontsize=9)
            ax.text(j+0.3, i+0.7, f"{param_4[i][j]:.2f}", ha='right', va='top', fontsize=9)


    # plt.title(f'{title} - Param 1 & Param 2')
    
    mytitle = 'NICE:priv:{}/distance:{}/plaus_distance:{}/correctness:{}'.format(nice_param[index_1],nice_param[index_2],nice_param[index_3],nice_param[index_4])
    plt.title(mytitle)
    plt.xlabel('Elsilon')
    plt.ylabel('k')
    plt.xticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons)
    plt.yticks(ticks=np.arange(len(ks)) + 0.5, labels=ks, rotation=0)
    
    plt.savefig(plot_file)
    plt.close()

        ########################################################################
    ############ Draw improvement heatmaps ############################
    ########################################################################
    imp_1 =  np.zeros((len(ks),len(epsilons)))
    imp_2 =  np.zeros((len(ks),len(epsilons)))
    imp_3 =  np.zeros((len(ks),len(epsilons)))
    imp_4 =  np.zeros((len(ks),len(epsilons)))
    for i in range(len(ks)):
        for j in range(len(epsilons)):
            if index_1 in higher_better:
                imp_1[i][j] = ((param_1[i][j] - nice_param[index_1]) / nice_param[index_1])*100
            else:
                imp_1[i][j] = ((nice_param[index_1] - param_1[i][j]) / nice_param[index_1])*100
            if index_2 in higher_better:
                imp_2[i][j] = ((param_2[i][j] - nice_param[index_2]) / nice_param[index_2])*100
            else:
                imp_2[i][j] = ((nice_param[index_2] - param_2[i][j]) / nice_param[index_2])*100
            if index_3 in higher_better:
                imp_3[i][j] = ((param_3[i][j] - nice_param[index_3]) / nice_param[index_3])*100
            else:
                imp_3[i][j] = ((nice_param[index_3] - param_3[i][j]) / nice_param[index_3])*100
            if index_4 in higher_better:
                imp_4[i][j] = ((param_4[i][j] - nice_param[index_4]) / nice_param[index_4])*100
            else:
                imp_4[i][j] = ((nice_param[index_4] - param_4[i][j]) / nice_param[index_4])*100


    plt.figure(figsize=(15,5))
    cbar_title = {'label': f'{title}  improvement'} 
    norm = plt.Normalize(vmin=imp_1.min(), vmax=imp_1.max())
    purples_palette = sns.light_palette("purple", as_cmap=True)
    blue_palette = sns.light_palette("blue", as_cmap=True)
    ax = sns.heatmap(imp_1,  cmap=blue_palette,  fmt='', norm = norm,cbar_kws=cbar_title)
    
    for i in range(len(ks)):
        for j in range(len(epsilons)):
            # Main value (param_1), larger and centered
            ax.text(j+.5, i+.5, f"{imp_1[i][j]:.2f}", ha='center', va='center', fontsize=11)
            # Secondary value (param_2), smaller and in the corners
            ax.text(j+0.8, i+0.2, f"{imp_2[i][j]:.1f}", ha='right', va='top', fontsize=11)
            ax.text(j+0.8, i+0.8, f"{imp_3[i][j]:.1f}", ha='right', va='top', fontsize=11)
            ax.text(j+0.2, i+0.8, f"{imp_4[i][j]:.1f}", ha='right', va='top', fontsize=11)



    
    # plt.title={'label': f'{title} improvement'}
    plt.xlabel('Epsilon values')
    plt.xticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons)
    plt.yticks(ticks=np.arange(len(ks)) + 0.5, labels=ks, rotation=0)
    plt.savefig(imp_plot_file)
    plt.close()

def draw_one_dimensional_heatmap_5param(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons, index_1, index_2 , index_3 ,index_4,index_5,nice_param,title,higher_better):
    # generate plot file path
    if RANDOM_SEED != 9999:  ### aggregated results
        plot_file_path = '{}/{}/{}/'.format(graphs_dir,RANDOM_SEED,method_name)
    else:
        plot_file_path = '{}/{}/'.format(graphs_dir,method_name)
    if not os.path.exists(plot_file_path):
        os.makedirs(plot_file_path, exist_ok=True)
    plot_file = '{}/{}.png'.format(plot_file_path,title)
    imp_plot_file = '{}/{}_improvement.png'.format(plot_file_path,title)

    
    param_1 = np.zeros(len(epsilons))
    param_2 = np.zeros(len(epsilons))
    param_3 = np.zeros(len(epsilons))
    param_4 = np.zeros(len(epsilons))
    param_5 = np.zeros(len(epsilons))
    for i in range(len(epsilons)):
        param_1[i] = CF_DF[i][index_1]
        param_2[i]= CF_DF[i][index_2]
        param_3[i]= CF_DF[i][index_3]
        param_4[i]= CF_DF[i][index_4]
        param_5[i]= CF_DF[i][index_5]
    # Create a 2D array for heatmap where each row is the param value repeated
    # This is a workaround to use seaborn's heatmap for one-dimensional data
    # combined_annotations = np.array([[f"{param_1[i]:.2f}\n{param_2[i]:.2f}"] for i in range(len(epsilons))])
    
    plt.figure(figsize=(8, 2))  # Adjusted figure size for one-dimensional data
    cbar_title = {'label': f'{title}'}
    purples_palette = sns.light_palette("purple", as_cmap=True)
    blues_palette = sns.light_palette("blue", as_cmap=True)
    if index_1 in higher_better:
        # sns.heatmap(param_1.reshape(-1,1), annot=combined_annotations, cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
        ax = sns.heatmap(param_1.reshape(1,-1), cmap=purples_palette,  fmt='', vmin=param_1.min(), vmax=param_1.max(),cbar_kws=cbar_title)
    else:
        # sns.heatmap(param_1.reshape(-1,1), annot=combined_annotations, cmap="Blues",  fmt='', vmin=param_1.max(), vmax=param_1.min())
        ax = sns.heatmap(param_1.reshape(1,-1),  cmap=blues_palette,  fmt='', vmin=param_1.max(), vmax=param_1.min(),cbar_kws=cbar_title)

    for i in range(len(epsilons)):
    
        # Main value (param_1), larger and centered
        ax.text(i+.3,.5, f"{param_1[i]:.2f}", ha='center', va='center', fontsize=11)
        # Secondary value (param_2), smaller and in the top-right corner
        # This is a workaround; precise positioning like top-right is limited
        ax.text( i+0.8, .3,f"{param_2[i]:.2f}", ha='right', va='top', fontsize=7)
        ax.text( i+0.8, .8,f"{param_3[i]:.2f}", ha='right', va='top', fontsize=7)
        ax.text( i+0.3, .8,f"{param_4[i]:.2f}", ha='right', va='top', fontsize=7)
        ax.text( i+0.8, .8,f"{param_5[i]:.2f}", ha='right', va='top', fontsize=7)


    # plt.title(f'{title} - Param')
    mytitle = 'NICE:priv:{}/distance:{}/plaus_distance:{}/correctness:{}'.format(nice_param[index_1],nice_param[index_2],nice_param[index_3],nice_param[index_4])
    plt.title(mytitle)
    plt.xlabel('Epsilon values')
    plt.xticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons)
    plt.savefig(plot_file)
    plt.close()
    
    ########################################################################
    ############ Draw improvement heatmaps ############################
    ########################################################################
    imp_1 = np.zeros(len(epsilons))
    imp_2 = np.zeros(len(epsilons))
    imp_3 = np.zeros(len(epsilons))
    imp_4 = np.zeros(len(epsilons))
    imp_5 = np.zeros(len(epsilons))
    for i in range(len(epsilons)):
        if index_1 in higher_better:
            imp_1[i] = ((param_1[i] - nice_param[index_1]) / ((nice_param[index_1])+0.000000000001))*100
        else:
            imp_1[i] = ((nice_param[index_1] - param_1[i]) / ((nice_param[index_1])+0.000000000001))*100
        if index_2 in higher_better:
            imp_2[i] = ((param_2[i] - nice_param[index_2]) / ((nice_param[index_2])+0.000000000001))*100
        else:
            imp_2[i] = ((nice_param[index_2] - param_2[i]) / ((nice_param[index_2])+0.000000000001))*100
        if index_3 in higher_better:
            imp_3[i] = ((param_3[i] - nice_param[index_3]) / ((nice_param[index_3])+0.000000000001))*100
        else:
            imp_3[i] = ((nice_param[index_3] - param_3[i]) / ((nice_param[index_3])+0.000000000001))*100
        if index_4 in higher_better:
            imp_4[i] = ((param_4[i] - nice_param[index_4]) / ((nice_param[index_4])+0.000000000001))*100
        else:
            imp_4[i] = ((nice_param[index_4] - param_4[i]) / ((nice_param[index_4])+0.000000000001))*100
        if index_5 in higher_better:
            imp_5[i] = ((param_5[i] - nice_param[index_5]) / ((nice_param[index_5])+0.000000000001))*100
        else:
            imp_5[i] = ((nice_param[index_5] - param_5[i]) / ((nice_param[index_5])+0.000000000001))*100

    plt.figure(figsize=(10, 2))  # Adjusted figure size for one-dimensional data
        # sns.heatmap(param_1.reshape(-1,1), annot=combined_annotations, cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
    cbar_title = {'label': f'{title}  improvement'} 
    norm = plt.Normalize(vmin=imp_1.min(), vmax=imp_1.max())
    purples_palette = sns.light_palette("purple", as_cmap=True)
    blues_palette = sns.light_palette("blue", as_cmap=True)
    ax = sns.heatmap(imp_1.reshape(1,-1), cmap=purples_palette,  fmt='', norm = norm,cbar_kws=cbar_title)
    def get_text_color(background_color):
        r, g, b, _ = background_color
        luminance = 0.299*r + 0.587*g + 0.114*b
        return 'white' if luminance < 0.5 else 'black'
    for i in range(len(epsilons)):
        background_color = purples_palette(norm(imp_1[i]))
        text_color = get_text_color(background_color)
        # Main value (param_1), larger and centered
        ax.text(i+.5,.5, f"{imp_1[i]:.2f}", ha='center', va='center', fontsize=11,color=text_color)
        # Secondary value (param_2), smaller and in the top-right corner
        ax.text( i+0.9, .2,f"{imp_2[i]:.1f}", ha='right', va='top', fontsize=11,color=text_color)
        ax.text( i+0.9, .7,f"{imp_3[i]:.1f}", ha='right', va='top', fontsize=11,color=text_color)
        ax.text( i+0.4, .7,f"{imp_4[i]:.1f}", ha='right', va='top', fontsize=11,color=text_color)
        ax.text( i+0.4, .2,f"{imp_5[i]:.1f}", ha='right', va='top', fontsize=11,color=text_color)


    # sns.heatmap(param_1, annot=combined_annotations, cmap="viridis", cbar_kws={'label': 'Param 1 Value'}, fmt='')
    # plt.title(f'{title} - Param')
    plt.cbar_kws={'label': f'{title} improvement'}
    plt.xlabel('Epsilon values')
    plt.xticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons)
    plt.savefig(imp_plot_file)
    plt.close()


def draw_two_dimensional_heatmap_5param(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks ,index_1, index_2 , index_3 ,index_4,index_5,nice_param,title,higher_better):
    # generate plot file path
    if RANDOM_SEED != 9999:  ### aggregated results
        plot_file_path = '{}/{}/{}/'.format(graphs_dir,RANDOM_SEED,method_name)
    else:
        plot_file_path = '{}/{}/'.format(graphs_dir,method_name)

    if not os.path.exists(plot_file_path):
        os.makedirs(plot_file_path, exist_ok=True)
    plot_file = '{}/{}.png'.format(plot_file_path,title)
    imp_plot_file = '{}/{}_improvement.png'.format(plot_file_path,title)
    # CF_DF is 2 dimensional array so, param_1 and param_2 are 2 dimensional arrays
    # param_1 = np.zeros((len(epsilons), len(ks)))
    # param_2 = np.zeros((len(epsilons), len(ks)))
    # param_3 = np.zeros((len(epsilons), len(ks)))
    # for i in range(len(epsilons)):
    #     for j in range(len(ks)):
    #         param_1[i][j] = CF_DF[i][j][index_1]
    #         param_2[i][j] = CF_DF[i][j][index_2]
    #         param_3[i][j] = CF_DF[i][j][index_3]
    param_1 = np.zeros((len(ks),len(epsilons)))
    param_2 = np.zeros((len(ks),len(epsilons)))
    param_3 = np.zeros((len(ks),len(epsilons)))
    param_4 = np.zeros((len(ks),len(epsilons)))
    param_5 = np.zeros((len(ks),len(epsilons)))
    for i in range(len(ks)):
        for j in range(len(epsilons)):
            param_1[i][j] = CF_DF[j][i][index_1]
            param_2[i][j] = CF_DF[j][i][index_2]
            param_3[i][j] = CF_DF[j][i][index_3]
            param_4[i][j] = CF_DF[j][i][index_4]
            param_5[i][j] = CF_DF[j][i][index_5]
    # combined_annotations = np.array([[f"{param_1[i][j]:.2f}\n{param_2[i][j]:.2f}" for j in range(len(ks))] for i in range(len(epsilons))])

    plt.figure(figsize=(16, 7))
    cbar_title = {'label': f'{title}'}
    purples_palette = sns.light_palette("purple", as_cmap=True)
    blue_palette = sns.light_palette("blue", as_cmap=True)
    if index_1 in higher_better:
        # sns.heatmap(param_1, annot=combined_annotations, cmap="Purples",  fmt='', vmin=param_1.min(), vmax=param_1.max())
        ax = sns.heatmap(param_1,  cmap=purples_palette,  fmt='', vmin=param_1.min(), vmax=param_1.max(),cbar_kws=cbar_title)
    else:
        # sns.heatmap(param_1, annot=combined_annotations, cmap="Blues",  fmt='', vmin=param_1.max(), vmax=param_1.min())
        ax = sns.heatmap(param_1, cmap=blue_palette,  fmt='', vmin=param_1.max(), vmax=param_1.min(),cbar_kws=cbar_title)
    
    
    for i in range(len(ks)):
        for j in range(len(epsilons)):
            # Main value (param_1), larger and centered
            ax.text(j+.5, i+.5, f"{param_1[i][j]:.2f}", ha='center', va='center', fontsize=11)
            # Secondary value (param_2), smaller and in the top-right corner
            # This is a workaround; precise positioning like top-right is limited
            ax.text(j+0.8, i+0.2, f"{param_2[i][j]:.2f}", ha='right', va='top', fontsize=11)
            ax.text(j+0.8, i+0.8, f"{param_3[i][j]:.2f}", ha='right', va='top', fontsize=11)
            ax.text(j+0.2, i+0.8, f"{param_4[i][j]:.2f}", ha='right', va='top', fontsize=11)
            ax.text(j+0.2, i+0.8, f"{param_4[i][j]:.2f}", ha='right', va='top', fontsize=11)
            ax.text(j+0.2, i+0.2, f"{param_5[i][j]:.2f}", ha='right', va='top', fontsize=11)
    # plt.title(f'{title} - Param 1 & Param 2')
    
    mytitle = 'NICE:priv:{}/distance:{}/plaus_distance:{}/correctness:{}'.format(nice_param[index_1],nice_param[index_2],nice_param[index_3],nice_param[index_4])
    plt.title(mytitle)
    plt.xlabel('Elsilon')
    plt.ylabel('k')
    plt.xticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons)
    plt.yticks(ticks=np.arange(len(ks)) + 0.5, labels=ks, rotation=0)
    
    plt.savefig(plot_file)
    plt.close()

        ########################################################################
    ############ Draw improvement heatmaps ############################
    ########################################################################
    imp_1 =  np.zeros((len(ks),len(epsilons)))
    imp_2 =  np.zeros((len(ks),len(epsilons)))
    imp_3 =  np.zeros((len(ks),len(epsilons)))
    imp_4 =  np.zeros((len(ks),len(epsilons)))
    imp_5 =  np.zeros((len(ks),len(epsilons)))
    for i in range(len(ks)):
        for j in range(len(epsilons)):
            if index_1 in higher_better:
                imp_1[i][j] = ((param_1[i][j] - nice_param[index_1]) / ((nice_param[index_1])+0.000000000001))*100
            else:
                imp_1[i][j] = ((nice_param[index_1] - param_1[i][j]) / ((nice_param[index_1])+0.000000000001))*100
            if index_2 in higher_better:
                imp_2[i][j] = ((param_2[i][j] - nice_param[index_2]) / ((nice_param[index_2])+0.000000000001))*100
            else:
                imp_2[i][j] = ((nice_param[index_2] - param_2[i][j]) / ((nice_param[index_2])+0.000000000001))*100
            if index_3 in higher_better:
                imp_3[i][j] = ((param_3[i][j] - nice_param[index_3]) / ((nice_param[index_3])+0.000000000001))*100
            else:
                imp_3[i][j] = ((nice_param[index_3] - param_3[i][j]) / ((nice_param[index_3])+0.000000000001))*100
            if index_4 in higher_better:
                imp_4[i][j] = ((param_4[i][j] - nice_param[index_4]) / ((nice_param[index_4])+0.000000000001))*100
            else:
                imp_4[i][j] = ((nice_param[index_4] - param_4[i][j]) / ((nice_param[index_4])+0.000000000001))*100
            if index_5 in higher_better:
                imp_5[i][j] = ((param_5[i][j] - nice_param[index_5]) / ((nice_param[index_5])+0.000000000001))*100
            else:
                imp_5[i][j] = ((nice_param[index_5] - param_5[i][j]) / ((nice_param[index_5])+0.000000000001))*100


    plt.figure(figsize=(16,7))
    cbar_title = {'label': f'{title}  improvement'} 
    norm = plt.Normalize(vmin=imp_1.min(), vmax=imp_1.max())
    purples_palette = sns.light_palette("purple", as_cmap=True)
    blue_palette = sns.light_palette("blue", as_cmap=True)
    ax = sns.heatmap(imp_1,  cmap=purples_palette,  fmt='', norm = norm,cbar_kws=cbar_title)
    
    def get_text_color(background_color):
        r, g, b, _ = background_color
        luminance = 0.299*r + 0.587*g + 0.114*b
        return 'white' if luminance < 0.5 else 'black'

    for i in range(len(ks)):
        for j in range(len(epsilons)):
            background_color = purples_palette(norm(imp_1[i][j]))
            text_color = get_text_color(background_color)
            # Main value (param_1), larger and centered
            ax.text(j+.5, i+.5, f"{imp_1[i][j]:.2f}", ha='center', va='center', fontsize=11 ,color=text_color)
            # Secondary value (param_2), smaller and in the corners
            ax.text(j+0.8, i+0.2, f"{imp_2[i][j]:.1f}", ha='right', va='top', fontsize=11,color=text_color)
            ax.text(j+0.8, i+0.8, f"{imp_3[i][j]:.1f}", ha='right', va='top', fontsize=11,color=text_color)
            ax.text(j+0.3, i+0.8, f"{imp_4[i][j]:.1f}", ha='right', va='top', fontsize=11,color=text_color)
            ax.text(j+0.3, i+0.2, f"{imp_5[i][j]:.1f}", ha='right', va='top', fontsize=11,color=text_color)



    
    # plt.title={'label': f'{title} improvement'}
    plt.xlabel('Epsilon values')
    plt.ylabel('K values')
    plt.xticks(ticks=np.arange(len(epsilons)) + 0.5, labels=epsilons)
    plt.yticks(ticks=np.arange(len(ks)) + 0.5, labels=ks, rotation=0)
    plt.savefig(imp_plot_file)
    plt.close()


def Draw_table(CF_DF,graphs_dir,RANDOM_SEED,epsilons = None,ks = None ,nice_param = None,method_name='Exp_prox_DP',higher_better = [2,7]):
    if epsilons is None and ks is None and method_name == 'NICE':
        # It is nice
        {
            # no graph, just show 
        }
    elif epsilons is not None and ks is None:
        draw_one_dimensional_table_new(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,  nice_param=nice_param)
        # draw_one_dimensional_table(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,  index_1 = 3 ,index_2= 0,index_3 = 5,index_4 =7,index_5=2,nice_param=nice_param,title= 'M$_{1}$',higher_better=higher_better)

    elif epsilons is not None and ks is not None:
        draw_two_dimentional_table_new(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks, nice_param=nice_param)
    
    # weather_df = pd.DataFrame(np.random.rand(10,2)*5,
    #                       index=pd.date_range(start="2021-01-01", periods=10),
    #                       columns=["Tokyo", "Beijing"])

    # df = pd.DataFrame({
    # "strings": ["Adam", "Mike"],
    # "ints": [1, 3],
    # "floats": [1.123, 1000.23]
    # })
    # df.style \
    # .format(precision=3, thousands=".", decimal=",") \
    # .format_index(str.upper, axis=1) \
    # .relabel_index(["row 1", "row 2"], axis=0)


def draw_one_dimensional_table_new(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,nice_param):
        # record['average_distance'], 0
        # record['average_reidentification_rate'], 1
        # record['not_matching_count'],  2
        # record['one_exact_match'],  3
        # record['average_CF_min_dist'],  4
        # record['average_CF_min_k_dist'],  5
        # record['average_CF_rand_k_dist'],  6
        # record['success_rate'],  7
        # record['Have_match_count']  8
        # record['std_distance'],  9
        # record['std_CF_k_min_dist'], 10 

    # generate plot file path
    if RANDOM_SEED != 9999:  ### aggregated results
        plot_file_path = '{}{}/{}/'.format(graphs_dir,RANDOM_SEED,method_name)
    else:
        plot_file_path = '{}{}/'.format(graphs_dir,method_name)
    if not os.path.exists(plot_file_path):
        os.makedirs(plot_file_path, exist_ok=True)
    result_file = '{}/result.tex'.format(plot_file_path)
    imp_result_file = '{}/result_improvement.tex'.format(plot_file_path)

    print(CF_DF)
    temp_df = [[0]*9 for _ in range(len(epsilons)+1)]
    # temp_df = np.zeros((len(epsilons)+1,7))
    # extract data from DF_CF in order to create table
    nice_temp = np.zeros(10)
    temp_df[0][0] = "NICE"
    temp_df[0][1] = 0
    temp_df[0][2] = nice_param[0]
    temp_df[0][3] = nice_param[9]
    temp_df[0][4] = nice_param[5]
    temp_df[0][5] = nice_param[10]
    temp_df[0][6] = nice_param[7]
    temp_df[0][7] = nice_param[2]
    temp_df[0][8] = nice_param[3]
    
    
    for i in range(len(epsilons)):
        temp_df[i+1][0] =  method_name 
        temp_df[i+1][1] = epsilons[i]
        temp_df[i+1][2] = CF_DF[i][0] # average_distance
        temp_df[i+1][3] = CF_DF[i][9] # std_distance
        temp_df[i+1][4] = CF_DF[i][5] # average_CF_min_k_dist
        temp_df[i+1][5] = CF_DF[i][10] # std_CF_k_min_dist
        temp_df[i+1][6] = CF_DF[i][7] # Correctness
        temp_df[i+1][7] = CF_DF[i][2] # not_matching_count
        temp_df[i+1][8] = CF_DF[i][3] # one_exact_match
        
        
    
    
    
    epsilons_with_header = np.insert(epsilons, 0, 0)

    result_df = pd.DataFrame(temp_df,
                          index=epsilons_with_header, # pd.date_range(start="2021-01-01", periods=10),
                          columns=["Method","Epsilon", "$Dist_{prox} \\downarrow$","$Var_{Dist}$","$Plaus_{prox} \\downarrow$","$Var_{plaus}$","Correctness $\\uparrow$","$M_0 \\uparrow$","$M_1 \\downarrow$"])
                                    
    # Draw table


# Apply the function to the DataFrame to format the columns
# Assume upward arrows mean "higher is better" and downward arrows mean "lower is better"
    result_df["$Dist_{prox} \\downarrow$"] = bold_max(result_df["$Dist_{prox} \\downarrow$"], is_max=False)
    result_df["$Plaus_{prox} \\downarrow$"] = bold_max(result_df["$Plaus_{prox} \\downarrow$"], is_max=False)
    result_df["Correctness $\\uparrow$"] = bold_max(result_df["Correctness $\\uparrow$"], is_max=True)
    result_df["$M_0 \\uparrow$"] = bold_max(result_df["$M_0 \\uparrow$"], is_max=True)
    result_df["$M_1 \\downarrow$"] = bold_max(result_df["$M_1 \\downarrow$"], is_max=False)

    result_df['Epsilon'] = result_df.apply(lambda row: '-' if row['Method'] == 'NICE' else row['Epsilon'], axis=1)
    result_df['Method'] = result_df['Method'].apply(lambda x: latex_representations.get(x, x))
    mthd = result_df['Method'].apply(lambda x: latex_representations.get(x, x))
    result_df['Method'] = result_df['Method'].mask(result_df['Method'].duplicated(), '')
    result_df['Epsilon'] = result_df['Epsilon'].mask(result_df['Epsilon'].duplicated(), '')
    # Draw table
    # latex_table = result_df.to_latex(index=False, escape=False,float_format="{:.2f}".format)
    # latex_table_hline = add_hlines(latex_table,method_name)

    # latex_table_hline = add_hlines(latex_table,method_name)
    # with open(result_file, 'w') as f:
    #     f.write(latex_table_hline)
        
    result_df["$Var_{Dist}$"] = result_df["$Var_{Dist}$"].round(2)
    result_df["$Var_{plaus}$"] = result_df["$Var_{plaus}$"].round(2)

    result_df["$Dist_{prox} \\downarrow$"] = result_df["$Dist_{prox} \\downarrow$"].astype(str) + " ± " + result_df["$Var_{Dist}$"].astype(str)
    result_df["$Plaus_{prox} \\downarrow$"] = result_df["$Plaus_{prox} \\downarrow$"].astype(str) + " ± " + result_df["$Var_{plaus}$"].astype(str)

    result_df.drop(columns=["$Var_{Dist}$", "$Var_{plaus}$"], inplace=True)

    latex_table = result_df.to_latex(index=False, escape=False,float_format="{:.2f}".format)
    
    caption = f"\\caption{{Privacy-utility trade-off for {mthd[1]}}}\n"
    wrapped_latex_table = (
    "\\begin{table}[H!]\n"
    "\\centering\n"
    + latex_table +
    caption +
    "\\end{table}\n"
    )
    with open(result_file, 'w') as f:
        f.write(wrapped_latex_table)
    

    print("result Table saved at: ",result_file)

                  
    # Create improvement table
    nice_temp = np.zeros(8)
    nice_temp[0] = 0
    nice_temp[1] = nice_param[0]
    nice_temp[2] = nice_param[9]
    nice_temp[3] = nice_param[5]
    nice_temp[4] = nice_param[10]
    
    nice_temp[5] = nice_param[7]
    nice_temp[6] = nice_param[2]
    nice_temp[7] = nice_param[3]
    higher_better = [4,6]
    imp_df = [[0]*8 for _ in range(len(epsilons))] #  np.zeros((len(epsilons),6))
    for i in range(len(epsilons)):
        imp_df[i][0] = epsilons[i]
        for j in range(1,8):
            if j in higher_better:
                imp_df[i][j] = ((temp_df[i+1][j+1] - nice_temp[j]) / ((nice_temp[j])+0.000000000001))*100
            else:
                imp_df[i][j] = ((nice_temp[j] - temp_df[i+1][j+1]) / ((nice_temp[j])+0.000000000001))*100

    imp_result_df = pd.DataFrame(imp_df,
                          index=epsilons, 
                          columns=["Epsilon", "$Dist_{prox} \\downarrow$","$Var_{Dist}$","$Plaus_{prox} \\downarrow$","$Var_{plaus}$","Correctness $\\uparrow$","$M_0 \\uparrow$","$M_1 \\downarrow$"])

    # Find bests
    imp_result_df["$Dist_{prox} \\downarrow$"] = bold_max(imp_result_df["$Dist_{prox} \\downarrow$"], is_max=True)
    imp_result_df["$Plaus_{prox} \\downarrow$"] = bold_max(imp_result_df["$Plaus_{prox} \\downarrow$"], is_max=True)
    imp_result_df["Correctness $\\uparrow$"] = bold_max(imp_result_df["Correctness $\\uparrow$"], is_max=True)
    imp_result_df["$M_0 \\uparrow$"] = bold_max(imp_result_df["$M_0 \\uparrow$"], is_max=True)
    imp_result_df["$M_1 \\downarrow$"] = bold_max(imp_result_df["$M_1 \\downarrow$"], is_max=True)
    
    
    
    
    imp_result_df['Epsilon'] = imp_result_df['Epsilon'].mask(imp_result_df['Epsilon'].duplicated(), '')

    imp_result_df["$Var_{Dist}$"] = imp_result_df["$Var_{Dist}$"].round(2)
    imp_result_df["$Var_{plaus}$"] = imp_result_df["$Var_{plaus}$"].round(2)
    
    imp_result_df["$Dist_{prox} \\downarrow$"] = imp_result_df["$Dist_{prox} \\downarrow$"].astype(str) + " ± " + imp_result_df["$Var_{Dist}$"].astype(str)
    imp_result_df["$Plaus_{prox} \\downarrow$"] = imp_result_df["$Plaus_{prox} \\downarrow$"].astype(str) + " ± " + imp_result_df["$Var_{plaus}$"].astype(str)

    imp_result_df.drop(columns=["$Var_{Dist}$", "$Var_{plaus}$"], inplace=True)

    imp_latex_table = imp_result_df.to_latex(index=False, escape=False,float_format="{:.2f}".format)
    imp_latex_table_hline = add_hlines(imp_latex_table,method_name)
    #with open(imp_result_file, 'w') as f:
    #    f.write(imp_latex_table_hline)
     
    # latex_table = result_df.to_latex(index=False, escape=False,float_format="{:.2f}".format)
    mthd = result_df['Method'].apply(lambda x: latex_representations.get(x, x))
    caption = f"\\caption{{Privacy-utility trade-off for {mthd[1]}}}\n"
    wrapped_latex_table = (
    "\\begin{table}[H!]\n"
    "\\centering\n"
    + imp_latex_table_hline +
    caption +
    "\\end{table}\n"
    )
    imp_result_file = '{}/result_improvement.tex'.format(plot_file_path)
    with open(imp_result_file, 'w') as f:
        f.write(wrapped_latex_table)
    
    
    print("improvement Table saved at: ",imp_result_file)
    


def draw_two_dimentional_table_new(CF_DF,graphs_dir,method_name,RANDOM_SEED, epsilons,ks,nice_param):
    if RANDOM_SEED != 9999:  ### aggregated results
        plot_file_path = '{}/{}/{}/'.format(graphs_dir,RANDOM_SEED,method_name)
    else:
        plot_file_path = '{}/{}/'.format(graphs_dir,method_name)

    if not os.path.exists(plot_file_path):
        os.makedirs(plot_file_path, exist_ok=True)
    result_file = '{}/results.tex'.format(plot_file_path)
    imp_result_file = '{}/result_improvement.tex'.format(plot_file_path)

    

    temp_df = [[0]*10 for _ in range((len(epsilons)*len(ks))+1)]
    # temp_df = np.zeros((len(epsilons)+1,7))
    # extract data from DF_CF in order to create table
    #first row is NICE
    nice_temp = np.zeros(8)
    temp_df[0][0] = "NICE"
    temp_df[0][1] = 0
    temp_df[0][2] = 1
    temp_df[0][3] = nice_param[0]
    temp_df[0][4] = nice_param[9]
    temp_df[0][5] = nice_param[5]
    temp_df[0][6] = nice_param[10]
    temp_df[0][7] = nice_param[7]
    temp_df[0][8] = nice_param[2]
    temp_df[0][9] = nice_param[3]
    
    for i in range(len(epsilons)):
        for j in range(len(ks)):
            temp_df[i*len(ks)+j+1][0] = method_name
            temp_df[i*len(ks)+j+1][1] = epsilons[i]
            temp_df[i*len(ks)+j+1][2] = ks[j]
            temp_df[i*len(ks)+j+1][3] = CF_DF[i][j][0] # average_distance
            temp_df[i*len(ks)+j+1][4] = CF_DF[i][j][9] # variance_distance
            temp_df[i*len(ks)+j+1][5] = CF_DF[i][j][5] # average_CF_min_k_dist
            temp_df[i*len(ks)+j+1][6] = CF_DF[i][j][10] # variance_CF_min_k _dist
            temp_df[i*len(ks)+j+1][7] = CF_DF[i][j][7] # Correctness
            temp_df[i*len(ks)+j+1][8] = CF_DF[i][j][2] # not_matching_count
            temp_df[i*len(ks)+j+1][9] = CF_DF[i][j][3] # one_exact_match
        # temp_df[i+1][0] = method_name
        # temp_df[i+1][1] = epsilons[i]
        # temp_df[i+1][2] = CF_DF[i][0] # average_distance
        # temp_df[i+1][3] = CF_DF[i][5] # average_CF_min_k_dist
        # temp_df[i+1][4] = CF_DF[i][7] # Correctness
        # temp_df[i+1][5] = CF_DF[i][2] # not_matching_count
        # temp_df[i+1][6] = CF_DF[i][3] # one_exact_match
    
    
    
    # epsilons_with_header = np.insert(epsilons, 0, 0)

    # columns = ["Method","Epsilon","K", "$Dist_{prox} \\downarrow$","$Plaus_{prox} \\downarrow$","Correctness $\\uparrow$","$M_0 \\uparrow$","$M_1 \\downarrow$"]
    columns=["Method","Epsilon","K", "$Dist_{prox} \\downarrow$","$Var_{Dist}$","$Plaus_{prox} \\downarrow$","$Var_{plaus}$","Correctness $\\uparrow$","$M_0 \\uparrow$","$M_1 \\downarrow$"]
    result_df = pd.DataFrame(temp_df, columns=columns)
    # result_df = pd.DataFrame(temp_df,
                        #   index=epsilons_with_header, # pd.date_range(start="2021-01-01", periods=10),
                        #   columns=["Method","epsilon","K", "$Dist_{prox} \\downarrow$","$plaus_{prox} \\downarrow$","Correctness $\\uparrow$","$M_0 \\uparrow$","$M_1 \\downarrow$"])

    result_df["$Dist_{prox} \\downarrow$"] = bold_max(result_df["$Dist_{prox} \\downarrow$"], is_max=False)
    result_df["$Plaus_{prox} \\downarrow$"] = bold_max(result_df["$Plaus_{prox} \\downarrow$"], is_max=False)
    result_df["Correctness $\\uparrow$"] = bold_max(result_df["Correctness $\\uparrow$"], is_max=True)
    result_df["$M_0 \\uparrow$"] = bold_max(result_df["$M_0 \\uparrow$"], is_max=True)
    result_df["$M_1 \\downarrow$"] = bold_max(result_df["$M_1 \\downarrow$"], is_max=False)
    result_df['Epsilon'] = result_df.apply(lambda row: '-' if row['Method'] == 'NICE' else row['Epsilon'], axis=1)                            
    result_df['Method'] = result_df['Method'].apply(lambda x: latex_representations.get(x, x))
    result_df['Method'] = result_df['Method'].mask(result_df['Method'].duplicated(), '')
    result_df['Epsilon'] = result_df['Epsilon'].mask(result_df['Epsilon'].duplicated(), '')
    
# update the table with the variance values
    result_df["$Var_{Dist}$"] = result_df["$Var_{Dist}$"].round(2)
    result_df["$Var_{plaus}$"] = result_df["$Var_{plaus}$"].round(2)

    result_df["$Dist_{prox} \\downarrow$"] = result_df["$Dist_{prox} \\downarrow$"].astype(str) + " ± " + result_df["$Var_{Dist}$"].astype(str)
    result_df["$Plaus_{prox} \\downarrow$"] = result_df["$Plaus_{prox} \\downarrow$"].astype(str) + " ± " + result_df["$Var_{plaus}$"].astype(str)

    result_df.drop(columns=["$Var_{Dist}$", "$Var_{plaus}$"], inplace=True)
    # Draw table
    latex_table = result_df.to_latex(index=False, escape=False,float_format="{:.2f}".format)
    latex_table_hline = add_hlines(latex_table,method_name)
    mthd = result_df['Method'].apply(lambda x: latex_representations.get(x, x))
    caption = f"\\caption{{Privacy-utility trade-off for {mthd[1]}}}\n"
    wrapped_latex_table = (
    "\\begin{table}[ht]\n"
    "\\centering\n"
    + latex_table_hline +
    caption +
    "\\end{table}\n"
    )
    with open(result_file, 'w') as f:
        f.write(wrapped_latex_table)

    # print(latex_table_hline)

        
    # Create improvement table
    nice_temp = np.zeros(9)
    nice_temp[0] = 0 #epsilon
    nice_temp[1] = 1 # k
    nice_temp[2] = nice_param[0]
    nice_temp[3] = nice_param[9]
    nice_temp[4] = nice_param[5]
    nice_temp[6] = nice_param[10]
    nice_temp[6] = nice_param[7]
    nice_temp[7] = nice_param[2]
    nice_temp[8] = nice_param[3]
    higher_better = [6,7]
    imp_df = [[0]*9 for _ in range((len(epsilons)*len(ks)))]
    # imp_df = [[0]*6 for _ in range(len(epsilons))] #  np.zeros((len(epsilons),6))
    for i in range(len(epsilons)):
        for k in range(len(ks)):
            imp_df[i*len(ks)+k][0] = epsilons[i]
            imp_df[i*len(ks)+k][1] = ks[k]

            # imp_df[k+(i*k)][0] = epsilons[i]
            for j in range(2,9):    # iterate over the columns
                if j in higher_better:
                    imp_df[k+(i*len(ks))][j] = ((temp_df[k+(i*len(ks))+1][j+1] - nice_temp[j]) / ((nice_temp[j])+0.000000000001))*100
                else:
                    imp_df[k+(i*len(ks))][j] = ((nice_temp[j] - temp_df[k+(i*len(ks))+1][j+1]) / ((nice_temp[j])+0.000000000001))*100

    imp_result_df = pd.DataFrame(imp_df,
                          columns=["Epsilon","K", "$Dist_{prox} \\downarrow$","$Var_{Dist}$","$Plaus_{prox} \\downarrow$","$Var_{plaus}$","Correctness $\\uparrow$","$M_0 \\uparrow$","$M_1 \\downarrow$"])
    # imp_result_df.set_index(['Epsilon', 'K'], inplace=True)
    # Find bests
    imp_result_df["$Dist_{prox} \\downarrow$"] = bold_max(imp_result_df["$Dist_{prox} \\downarrow$"], is_max=True)
    imp_result_df["$Plaus_{prox} \\downarrow$"] = bold_max(imp_result_df["$Plaus_{prox} \\downarrow$"], is_max=True)
    imp_result_df["Correctness $\\uparrow$"] = bold_max(imp_result_df["Correctness $\\uparrow$"], is_max=True)
    imp_result_df["$M_0 \\uparrow$"] = bold_max(imp_result_df["$M_0 \\uparrow$"], is_max=True)
    imp_result_df["$M_1 \\downarrow$"] = bold_max(imp_result_df["$M_1 \\downarrow$"], is_max=True)
    
    # imp_result_df['Method'] = imp_result_df['Method'].apply(lambda x: latex_representations.get(x, x))
    # imp_result_df['Method'] = imp_result_df['Method'].mask(imp_result_df['Method'].duplicated(), '')
    imp_result_df['Epsilon'] = imp_result_df['Epsilon'].mask(imp_result_df['Epsilon'].duplicated(), '')


    imp_result_df['Epsilon'] = imp_result_df['Epsilon'].mask(imp_result_df['Epsilon'].duplicated(), '')

    imp_result_df["$Var_{Dist}$"] = imp_result_df["$Var_{Dist}$"].round(2)
    imp_result_df["$Var_{plaus}$"] = imp_result_df["$Var_{plaus}$"].round(2)
    
    imp_result_df["$Dist_{prox} \\downarrow$"] = imp_result_df["$Dist_{prox} \\downarrow$"].astype(str) + " ± " + imp_result_df["$Var_{Dist}$"].astype(str)
    imp_result_df["$Plaus_{prox} \\downarrow$"] = imp_result_df["$Plaus_{prox} \\downarrow$"].astype(str) + " ± " + imp_result_df["$Var_{plaus}$"].astype(str)

    imp_result_df.drop(columns=["$Var_{Dist}$", "$Var_{plaus}$"], inplace=True)


    imp_latex_table = imp_result_df.to_latex(index=False, escape=False,float_format="{:.2f}".format)
    imp_latex_table_hline = add_hlines(imp_latex_table,method_name)
    
    mthd = result_df['Method'].apply(lambda x: latex_representations.get(x, x))
    caption = f"\\caption{{Privacy-utility trade-off improvement for {mthd[1]}}}\n"
    wrapped_latex_table = (
    "\\begin{table}[ht]\n"
    "\\centering\n"
    + imp_latex_table_hline +
    caption +
    "\\end{table}\n"
    )
    
    with open(imp_result_file, 'w') as f:
        f.write(wrapped_latex_table)
    

    print("Improvement Table saved at: ",imp_result_file)

    # imp_result_df.to_latex(plot_file_path+'result_improvement_table.tex',index=False, float_format="{:.2f}".format)
    # plt.close()
    
                                
    
    # Draw table
    # latex_table = result_df.to_latex(index=False, escape=False,float_format="{:.2f}".format)
    # latex_table_hline = add_hlines(latex_table,method_name)
    # with open(result_file, 'w') as f:
    #     f.write(latex_table_hline)
    # print("Improvement Table saved at: ",plot_file_path)
    
    
    
    
    