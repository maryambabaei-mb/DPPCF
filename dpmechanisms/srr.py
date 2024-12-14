import numpy as np
import pandas as pd
import random
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import math

#from utils.analysis.draw_plots import plot_number_frequency

def plot_number_frequency(value_list,title='Statistical_analysis',input=0):
    # Count the frequency of each number
    frequency = Counter(value_list)
    
    # Extract numbers and corresponding frequencies
    numbers = list(frequency.keys())
    counts = list(frequency.values())
    
    # Plot the data
    plt.bar(numbers, counts, color='blue')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title + "for input :" +str(input))
    plt.show()
    
class MinMaxItem:   # this class is being used to save min and max values of each group in SRR for numercal features
    def __init__(self, min_val = None,max_val = None):
        self.min = min_val
        self.max = max_val


# d is domain size, so for each feature value we have a different domain size
# for now, for numerical values we consider d equal to the sensitivity. maybe we should update it
def find_optimum_C_and_m(epsilon,d):
    
    # m is number of groups
    # c is ratio of alpha_max / alpha_min (alpha_i is the probability defined for each group)
    c = np.exp(epsilon)  # based on the paper
    m = (2 * (c * d - np.exp(np.log(c)+1))) / ((c-1) * d) 
    # return m and c
    return round(m),c

# generates the probalilies associated to each value group based on their distance to the input
def generate_alphas(m,c,alpha_min):
    # consider we have m and c, find alphas which will conduct the random selection mechanism
    # calculate alphas
    alphas = []
    #alphas.append(alpha_min)
    delta = ( alpha_min * (c-1) ) / (m -1)
    alpha_p = alpha_min
    for i in range(m):
        alphas.append(alpha_p)
        alpha_p = alpha_p + delta
        
    ####### I think we should reverse the alphas to give highiest probalility to the closest group
    descending_alphas = alphas[::-1]
    return descending_alphas

# get the minimum and maximum values of an interval of numerical values and adds them to the associated group
def insert_interval(min,max,group_list):
    interval = MinMaxItem(min,max)
    group_list.append(interval)
    return group_list

#gets the selected group and randomly retuen a value belonging to one of its intervals
def random_selection(group_list):
    interval_count = len(group_list)
    if (interval_count == 2):
        #randomly select one interval
        prob = random.random()
        if(prob > .5):
            interval = group_list[1]
        else:
            interval = group_list[0]    
    # either prob < .5 or list only has one interval
    else:
        interval = group_list[0]
         # then it should be 1
        # select the only interval
    
    # rendomly select one value from the selected interval
    return random.randint(interval.min,interval.max) #selecte_value


def Generate_num_groups(input,m,min_val,max_val):
    # consider we have m and c, find alphas and select LDP values for categorical features
    # question: what if number of categories is less than m? what should we do to solve the problem?
        ##### generate groups
    #each group is a list of categorical frature values
    group_list = [[] for _ in range(m)]
    group_size = (max_val - min_val) / m
    
    #find the range of values in each group
    # point: we need to group values based on their distance to input
    # start generating 
    min1 = input
    max2 = input
    no_left = False
    no_right = False 
    for i in range(m): # start from closest values to input, by adding/decrising to it find possible values for each group
        if (min1 + round(group_size/2) < max_val and max2-round(group_size/2) > min_val):
            max1 = round(min1 + round(group_size/2))
            min2 = round(max2 - round(group_size/2))
        elif (min1 + round(group_size/2) < max_val and max2-round(group_size/2) <= min_val and max2 > min_val):
            min2 = min_val
            remain_size = round(group_size) - (max2-min2)
            max1 = min(max_val,min1 + remain_size)
        elif (min1 + round(group_size/2) >= max_val and max2-round(group_size/2) > min_val  and min1 < max_val):
            max1 = max_val
            remain_size = round(group_size) - (max1-min1)
            min2 = max(min_val,max2 - remain_size)

        elif(min1 < max_val and min1 + round(group_size/2) >= max_val and max2 > min_val and max2-round(group_size/2) <= min_val):
            min2 = min_val
            max1 = max_val
            remain_size = 0

        elif(min1 >= max_val):
            #just go to left stop changing rightside
            min2 = max(max2 - round(group_size),min_val)
            no_right = True
        elif(max2 <= min_val): 
            #just go to right
            max1 = min(min1+round(group_size),max_val)
            no_left = True
        
        
        if(min1 == max2):
            #add one interval including min2 and max1
            group_list[i] = insert_interval(min2,max1,group_list[i])
        elif(no_left):
            max2 = min(max_val,round(min2+group_size))
            group_list[i] = insert_interval(min1,max1,group_list[i])  # the interval in the right side of the input is added last
        elif(no_right):
            min1 = max(min_val,round(max1 - group_size))
            group_list[i] = insert_interval(min2,max2,group_list[i])  # the interval in the left side of the input is added first
        else:
            # add both intervals to the group
            group_list[i] = insert_interval(min2,max2,group_list[i])  # the interval in the left side of the input is added first
            group_list[i] = insert_interval(min1,max1,group_list[i])  # the interval in the right side of the input is added last
        
        min1 = max1 + 1
        max2 = min2 - 1

    return group_list

def Get_SRR_num_feature(m,c,alpha_min,group_list):
    
    if m>1:
        alphas = generate_alphas(m,c,alpha_min)    
    else:
        alphas = [1]
    #alphas = generate_alphas(m,c,alpha_min)    
    #group_list = Generate_num_groups(input,m,min_val,max_val)    
    # now we have the probabilities and list of vaues in each group, we should choose the SRR using this information
    

    prob = random.random()
    group_index =m-1
    
    # for p in alphas: # the alpha array should be sorted based on how we generate it, find the group from which the value should be selected (the first group is closest to the input and has the highest probabity for selection)
    #     if(prob > p): 
    #         break
    #     else:
    #         group_index += 1
    while(group_index > 0 and alphas[group_index] < prob):
        group_index -= 1

    # the group is selected, we need to fetch its interval and randomly select one value from that interval
    
    selected_group = group_list[group_index]
    return random_selection(selected_group)

def generate_cat_group(input,feat_vals,m):
    differences = [(abs(x - input), x) for x in feat_vals]

    # Sort by differences
    differences.sort()
    differences.remove((0,input))
    # Calculate the size of each group
    group_size = len(feat_vals) // m

    # Divide the sorted list into m groups
    groups = [differences[i * group_size:(i + 1) * group_size] for i in range(m)]

    # If there are any remaining elements, distribute them evenly among the groups
    remaining = differences[m * group_size:]
    for i, (diff, value) in enumerate(remaining):
        groups[i % m].append((diff, value))

    # Extract values from groups
    grouped_values = [[value for _, value in group if value != input] for group in groups]

    return grouped_values



def Get_SRR_cat_feature(m,c,alpha_min,group_list):

    if m>1:
        alphas = generate_alphas(m,c,alpha_min)    
    else:
        alphas = [1]
    # group_list = generate_cat_group(input,feat_vals,m)    
   
    # now we have the probabilities and list of values in each group, we should choose the SRR using this information
    prob = random.random()
    group_index =m-1
    while(group_index > 0 and alphas[group_index] < prob):
        group_index -= 1
    # choose one random value from the selected group

    return np.random.choice(group_list[group_index])
    #return


def  Calc_alpha_min(m,c,d,grouplist,is_categorical):
    temp = 0
    if  is_categorical:
        for j in range(0,m-1):
            temp = temp + ((j) * len(grouplist[j]))
    
        alpha_min = (m-1) / (((m-1) * d * c)-(c-1)* temp)  
        #### Count size of the last group
        return alpha_min * len(grouplist[m-1]) 

    else:  #### numerical
        for j in range(0,m-1):
            if(len(grouplist[j]) ==1):
                g_size = grouplist[j][0].max - grouplist[j][0].min
            else:
                g_size = 0
                for i in range(len(grouplist[j])):
                    g_size += (grouplist[i][0].max - grouplist[i][0].min)
            temp = temp + ((j) * g_size)
    
        alpha_min = (m-1) / (((m-1) * d * c)-(c-1)* temp)   
        #### Count size of the last group
        if(len(grouplist[m-1]) ==1):
            g_size = grouplist[m-1][0].max - grouplist[m-1][0].min
        else:
            g_size = 0
            for i in range(len(grouplist[m-1])):
                g_size += (grouplist[i][0].max - grouplist[i][0].min)
    
    
        return alpha_min * g_size


def create_KDree():

    # Assuming 'data' is your dataset
    # Load your dataset into 'data' DataFrame
    data = pd.read_csv('adult_dataset.csv')

    # Assuming 'features' are the columns you want to consider for building the k-d tree
    # Select features accordingly
    features = data[['age', 'education-num', 'capital-gain', 'capital-loss']]

    # Build KDTree
    kdtree = KDTree(features, leaf_size=30)

    # Accessing the indices of nearest neighbors
    # Here, we're finding 5 nearest neighbors for the first data point
    distances, indices = kdtree.query(features.iloc[0].values.reshape(1, -1), k=5)

    print("Nearest neighbors indices:", indices)
    print("Distances to nearest neighbors:", distances)


def SRR_update(input,min_val,max_val,epsilon):
    valuelist =[]
    distancelist = []
    for i in range(int(max_val)-int(min_val)):
        if(min_val+i != input):
            distancelist.append(abs(int(min_val)+i - int(input)))
            valuelist.append(int(min_val)+i)
        # else:
        #     distancelist.append(0.0000001)


    scorelist = [1 / distance for distance in distancelist]
    # distance betwween values and input
    # score : opposit to distance
    # probability: based on score
    # score/some of the scores = probability
    probabilities = []
    for score in scorelist:
        probabilities.append(math.exp(epsilon * score))
    
        # Normalize probabilities
    total_prob = sum(probabilities)
    probabilities = [prob / total_prob for prob in probabilities]

    # Select item based on probabilities
    selected_index = random.choices(range(len(scorelist)), probabilities)[0]
    selected_item = valuelist[selected_index]
    return selected_item



if __name__ == '__main__':

    # test if it works
    #consider having epsilon and d, generate all other values
    num_input = 50
    min_num = 12
    max_num = 93
    epsilon = .5
    d_num = max_num - min_num
    cat_input = 12
    featur_values = {2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}
    d_cat = len(featur_values)
    is_categorical = False
    # 1- generate m and c
    
    #m,c = find_optimum_C_and_m(epsilon,len(feat_vals))  # when call in implementation
    #new_feature_value = get_ldp_cat_feature(x,feat_vals,epsilon,cat_type)


    # Generate groups to be able to find sum over them
    # if is_categorical:
    #     m,c = find_optimum_C_and_m(epsilon,d_cat)
    #     grouplist= generate_cat_group(cat_input,featur_values,m)
    #     if m>= 1:
    #         alpha_min = Calc_alpha_min(m,c,d_cat,grouplist,is_categorical)
    #     else:
    #         alpha_min = 1 
    # else: 
    #     m,c = find_optimum_C_and_m(epsilon,d_num)
    #     grouplist=Generate_num_groups(num_input,m,min_num,max_num)
    #     # 2- generate min_alpha
    #     if m>= 1:
    #         alpha_min = Calc_alpha_min(m,c,d_num,grouplist,is_categorical)
    #     else:
    #         alpha_min = 1
       
    
    
    
    updated_val = []
    if is_categorical:
        for i in range(100):
        # 4- Generate cat SRR
            new_feature_value = random.choice(list(featur_values))#Get_SRR_cat_feature(m,c,alpha_min,grouplist)
            print("updated value:",new_feature_value)
            updated_val.append(new_feature_value)
        # 5- if everything works generate an statistical analysis
        plot_number_frequency(updated_val,"categorical",cat_input)
    else:
        for i in range(100):
            # 3- Generate num SRR
    
            new_feature_value = SRR_update(num_input,min_num,max_num,epsilon)#Get_SRR_num_feature(m,c,alpha_min,grouplist)
            print("updated value:",new_feature_value)
            updated_val.append(new_feature_value)
        # 5- if everything works generate an statistical analysis
        plot_number_frequency(updated_val,"numerical",num_input)
        

    print("update_list",updated_val)
    #statistical analysis