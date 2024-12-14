import numpy as np
import pandas as pd
import sys
import copy
import matplotlib.pyplot as plt
import random
from dpmechanisms.srr import SRR_update  #Get_SRR_cat_feature,Get_SRR_num_feature,find_optimum_C_and_m

alpha_min = .2 #### Find a way to select optimum value

def get_ldp_num_feature(x,min_val,max_val,epsilon):
    #x = 0 # assuming user value ranges in [-1, 1], for example
    #sensitivity = 2 # maximum change a user will cause given range in [-1, 1]
    #epsilon = 1
    sensitivity = max_val - min_val
    # check if it is not considered privacy leakage
    y = x + np.random.laplace(loc=0, scale=sensitivity/epsilon)
    if y>max_val:
        return max_val
    elif y<min_val:
        return min_val
    else:    
        return int(y)

def get_ldp_cat_feature_value(x,feat_vals,epsilon):
    #### consider we talked about applying randomness on bits, not the whole value - so maybe we should implement local dp in a bitwise manner
    p =  1 / (1 + np.exp(epsilon))
    prob = random.random()
    
    if(prob < p):
        return np.random.choice(list(feat_vals))
    else:
        return x
    



def get_ldp_cat_feature(input,feat_vals,epsilon,cat_type=1):
    #cat_type = 1 means valuewise ldp for categorical features
    #cat_type = 0 means bitewise ldp for categorical features
    ##remove the input from the list
    r_feat_vals = feat_vals.copy()
    r_feat_vals.remove(input)
    p =  1 / (1 + np.exp(epsilon))    # the probability of flipping the value
    if(cat_type == 0):  #bit-wise
        result = 999999
        
        #changed = False
        # while ((result not in feat_vals) or changed ==False):
        result = 0
    #    temp = result
        mask = 0
        x = int(input)
        while x > 0:
            bit = x & 1  # Get the rightmost bit of x
            if random.random() < p:
                bit = 1 - bit  # Flip the bit with probability p
            result |= (bit << mask)  # Set the corresponding bit in the result
            x >>= 1  # Move to the next bit of x
            mask += 1  # Move the mask to the next bit position
            # if temp != result:
            #     changed = True
        if result in r_feat_vals:
            return result
        else:
            #edit result : if it is out of feature value list, change it to a better value
            return find_closest_val(r_feat_vals,result)
    else:   #value-wise
        prob = random.random()
    
        if(prob < p):
            return np.random.choice(list(r_feat_vals))
        else:
            return input



def Make_Private(CF,orig_instance,feature_ranges,cat_features,num_featuers,epsilon,dataset = None ,cat_type=1,ldp_Method = 0):   # srr = False
    # ldp_Method dictionary: 
    #     '0': LDP
    #     '1': SRR
    #     '2' : Noisy_max
        #cat_type = 1 means valuewise ldp for categorical features
    #cat_type = 0 means bitewise ldp for categorical features
    ldp_CF = CF.copy()
    #find different features between original instance and counterfactual
    nonoverlapping_features = np.where(orig_instance != CF, 1, 0) 
    #nonoverlapping_features = np.where(orig_instance != CF)[1]
    categorical_values = feature_ranges['categorical']
    numerical_min_values = feature_ranges['numerical']['min']
    numerical_max_values = feature_ranges['numerical']['max']

    
    #print(f"Numerical Feature Index {idx}: Min = {min_val}, Max = {max_val}")
    #find if different features are categorical or numerical
    #apply related local_dp on each feature value
    if np.sum(nonoverlapping_features[0]) > 0:
        epsilon = epsilon / np.sum(nonoverlapping_features[0])
    for i in range(len(nonoverlapping_features[0])):
        if nonoverlapping_features[0,i] == 1:
            x = CF[0,i]#the value of the feature in CF
            #if i is in categoricals: 
            if(i in cat_features):
            
                for idx, values in categorical_values.items():
                    if idx == i:
                        feat_vals = values
                        #find the list of possible values of the feature
                        # if(srr == False): 
                        ### for categorical features both localdp and our srr work in the same way
                        new_feature_value = get_ldp_cat_feature(x,feat_vals,epsilon,cat_type)           # pass to the function the possible values of categorical feature
                        # else:
                        #     m,c = find_optimum_C_and_m(epsilon,len(feat_vals))
                        #     new_feature_value = Get_SRR_cat_feature(input,feat_vals,m,c,alpha_min)           # pass to the function the possible values of categorical feature
                        break   # no need to iterate over other feature values
                
            
            #if i is numerical:
            else:
                for idx, (min_val, max_val) in enumerate(zip(numerical_min_values, numerical_max_values)):
                    if(num_featuers[idx] == i): 
                        #find the min and max value of the feature
                        if(ldp_Method == 0):   #ldp
                            new_feature_value = get_ldp_num_feature(x,min_val,max_val,epsilon)        # pass to the function max and min values of the numerical_feature
                        elif (ldp_Method ==1): #SRR
                        #     m,c = find_optimum_C_and_m(epsilon,max_val - min_val)
                            new_feature_value = SRR_update(x,min_val,max_val,epsilon) #Get_SRR_num_feature(input,min_val,max_val,)        # pass to the function max and min values of the numerical_feature
                        elif(ldp_Method == 2):
                            if(dataset is None):
                                print('No candidate list provided')
                                sys.exit()
                            else:
                                # feature_range = extract from feature_ranges
                                new_feature_value = get_noisy_max(dataset,i,feature_ranges,epsilon)
                                # LDP_using_noisy_max(orig_instance, CF, dataset , feature_ranges,epsilon)  ### here we pass indice only, we have divided epsilon, 
                        # break
            ldp_CF[0,i] = new_feature_value
        #else:
            #do not change 
            # the value in ldp-CF becuase it has not been changed
      
    
    return ldp_CF


def Compare(array1,array2):
    diff = np.where(array1 != array2)[1] 
    if len(diff) > 0:
        return 0
    else:
        return 1


def find_closest_val(features_list , input):# if DP value is not in the list, find the closest value of the feature list for it
    diff = 99999
    closest = input
    for value in features_list:
        
        if((value - input) < diff ):
            closest = value
            diff = value - input

    return closest

def find_feature_sensitivity(dataset,indice):
    {}

def get_value_frequency_pair(dataset,indice,sensitivity,epsilon):
        # the frequency should be noisy
        frequencies = []
        values = []
        feature_frequency = {}
        for sample in dataset:
            feature_value = sample[indice]

            # Update the frequency of this value for this feature
            feature_value = sample[indice]

            # Update the frequency of this value for this feature
            if feature_value in feature_frequency:
                feature_frequency[feature_value] += 1
            else:
                feature_frequency[feature_value] = 1
        
        frequencies.extend(list((feature_frequency.values())))
        values.extend(list((feature_frequency.keys())))
        epsilon = epsilon / len(frequencies)   ### To guarantee epsilon privacy we need to devide epsilon by size of R 
        for i in range(len(frequencies)):
            frequencies[i] = frequencies[i] + np.random.laplace(loc=0, scale=sensitivity/epsilon) #(sensitivity / epsilon)
        
        return values,frequencies

def get_noisy_max(dataset,indice,feature_range,epsilon):
        sensitivity = 1.0 #  For frequency function, sensitivity is 1
        value_list,frquency_list = get_value_frequency_pair(dataset,indice,sensitivity,epsilon)
        noisy_max_indice = np.argmax(frquency_list)
        return value_list[noisy_max_indice]
        # for each possible value for the feature:
        #     utility = frquency + laplace (delta u(sensitivity) / epsilon)
        # among these noisy frequencies select the maximom

def add_ldp_noise(Neighbour_list,diff,feature_ranges,epsilon,dataset=None,ldp_mech=0,cat_type =0) :

    categorical_values = feature_ranges['categorical']
    numerical_min_values = feature_ranges['numerical']['min']
    numerical_max_values = feature_ranges['numerical']['max']
    # 1. check the length of diff (number of ones in diff), and using K (based on length of ldp_NN and length of diff update epsilone)
    #Generate a list including indexes of numerical features
    num_featuers =[]
    
    for i in range(Neighbour_list[0].shape[1]):
        if i not in (feature_ranges['categorical']):
            num_featuers.append(i)
            
    if len(diff) > 0:
        epsilon = epsilon / (len(diff)*len(Neighbour_list))
    # 2. for each instance in neighbourlist update feat_val for positions in diff using LDP_mechanisms
    # 3. return updated list
    ldp_list = []
    for instance in Neighbour_list:
        for i in (diff):
            x = instance[0,i]#the value of the feature in CF
            #if i is in categoricals: 
            if(i in feature_ranges['categorical']):
            
                for idx, values in categorical_values.items():
                    if idx == i:
                        feat_vals = values
                        #find the list of possible values of the feature
                        # if(srr == False): 
                        ### for categorical features both localdp and our srr work in the same way
                        new_feature_value = get_ldp_cat_feature(x,feat_vals,epsilon,cat_type)           # pass to the function the possible values of categorical feature
                        # else:
                        #     m,c = find_optimum_C_and_m(epsilon,len(feat_vals))
                        #     new_feature_value = Get_SRR_cat_feature(input,feat_vals,m,c,alpha_min)           # pass to the function the possible values of categorical feature
                        break   # no need to iterate over other feature values
                
            
            #if i is numerical:
            else:
                for idx, (min_val, max_val) in enumerate(zip(numerical_min_values, numerical_max_values)):
                    if(num_featuers[idx] == i): 
                        #find the min and max value of the feature
                        if(ldp_mech == 0):   #ldp
                            new_feature_value = get_ldp_num_feature(x,min_val,max_val,epsilon)        # pass to the function max and min values of the numerical_feature
                        elif (ldp_mech ==1): #SRR
                        #     m,c = find_optimum_C_and_m(epsilon,max_val - min_val)
                            new_feature_value = SRR_update(x,min_val,max_val,epsilon) #Get_SRR_num_feature(input,min_val,max_val,)        # pass to the function max and min values of the numerical_feature
                        elif(ldp_mech == 2):
                            if(dataset is None):
                                print('No candidate list provided')
                                sys.exit()
                            else:
                                # feature_range = extract from feature_ranges
                                new_feature_value = get_noisy_max(dataset,i,feature_ranges,epsilon)
                                # LDP_using_noisy_max(orig_instance, CF, dataset , feature_ranges,epsilon)  ### here we pass indice only, we have divided epsilon, 
                        # break
            instance[0,i] = new_feature_value
        ldp_list.append(instance)
    return ldp_list

# ldp_DS = make_ldp(X_train,epsilon)
                # Make_Private(CF,to_explain,DS.dataset['feature_ranges'],DS.dataset['categorical_features'],DS.dataset['numerical_features'],epsilon,dataset=NICE_model.data.candidates_view,cat_type=1,ldp_Method=2)
def make_DS_LDP(Dataset,feature_ranges,epsilon,ldp_mech=0,cat_type =0) :
    feature_count = Dataset[0].shape[0]
    categorical_values = feature_ranges['categorical']
    numerical_min_values = feature_ranges['numerical']['min']
    numerical_max_values = feature_ranges['numerical']['max']
    # 1. check the length of diff (number of ones in diff), and using K (based on length of ldp_NN and length of diff update epsilone)
    #Generate a list including indexes of numerical features
    num_featuers =[]
    
    for i in range(Dataset[0].shape[0]):
        if i not in (feature_ranges['categorical']):
            num_featuers.append(i)
            
    # if len(diff) > 0:
        # epsilon = epsilon / (len(diff)*len(Dataset))
    epsilon = epsilon / feature_count
    # 2. for each instance in neighbourlist update feat_val for positions in diff using LDP_mechanisms
    # 3. return updated list
    LDP_Dataset = copy.deepcopy(Dataset)
    for ins in range(LDP_Dataset.shape[0]):
        instance = LDP_Dataset[ins]
        for i in (range(feature_count)):
            x = instance[i]#the value of the feature in CF
            #if i is in categoricals: 
            if(i in feature_ranges['categorical']):
            
                for idx, values in categorical_values.items():
                    if idx == i:
                        feat_vals = values
                        #find the list of possible values of the feature
                        # if(srr == False): 
                        ### for categorical features both localdp and our srr work in the same way
                        new_feature_value = get_ldp_cat_feature(x,feat_vals,epsilon,cat_type)           # pass to the function the possible values of categorical feature
                        # else:
                        #     m,c = find_optimum_C_and_m(epsilon,len(feat_vals))
                        #     new_feature_value = Get_SRR_cat_feature(input,feat_vals,m,c,alpha_min)           # pass to the function the possible values of categorical feature
                        break   # no need to iterate over other feature values
                
            
            #if i is numerical:
            else:
                for idx, (min_val, max_val) in enumerate(zip(numerical_min_values, numerical_max_values)):
                    if(num_featuers[idx] == i): 
                        #find the min and max value of the feature
                        if(ldp_mech == 0):   #ldp
                            new_feature_value = get_ldp_num_feature(x,min_val,max_val,epsilon)        # pass to the function max and min values of the numerical_feature
                        elif (ldp_mech ==1): #SRR
                        #     m,c = find_optimum_C_and_m(epsilon,max_val - min_val)
                            new_feature_value = SRR_update(x,min_val,max_val,epsilon) #Get_SRR_num_feature(input,min_val,max_val,)        # pass to the function max and min values of the numerical_feature
                        # elif(ldp_mech == 2):
                        #     if(dataset is None):
                        #         print('No candidate list provided')
                        #         sys.exit()
                        #     else:
                        #         # feature_range = extract from feature_ranges
                        #         new_feature_value = get_noisy_max(dataset,i,feature_ranges,epsilon)
                        #         # LDP_using_noisy_max(orig_instance, CF, dataset , feature_ranges,epsilon)  ### here we pass indice only, we have divided epsilon, 
                        break
            instance[i] = new_feature_value
        LDP_Dataset[ins] = instance
        # ldp_list.append(instance)

    return LDP_Dataset

# def LDP_using_noisy_max(orig_instance, nice_cf, candidate_list,feature_ranges,epsilon):

#         priv_cf = nice_cf.copy
#         nonoverlapping_features = np.where(orig_instance != nice_cf, 1, 0) 
#         # divide epsilon to make sure privacy budget is guaranteed
#         if np.sum(nonoverlapping_features[0]) > 0:
#             epsilon = epsilon / np.sum(nonoverlapping_features[0])
#         # for nonoverlapping features
#         for i in range(len(nonoverlapping_features[0])):    
#             if nonoverlapping_features[0,i] == 1:
#         feature_range = 0 
#         # extract featur_range
#         x = nice_cf[0,indice]#the value of the feature in CF
#         #### should we consider the probability of change, like in LDP?
#         priv_cf[0,i] = get_noisy_max(candidate_list,i,feature_range,epsilon)
        
#             # find max_frequency (or we already have it?)  # considering all members of tDS, candidate set? or nearest neighbours?
#         # based on sensitivity add noise and return value
#         # if it is categorical -> find other ways for adding noise



# def test_lsrr(CF,orig_instance,feature_ranges,cat_features,num_featuers,epsilon):
    
    
#     #cat_type = 1 means valuewise ldp for categorical features
#     #cat_type = 0 means bitewise ldp for categorical features
#     ldp_CF = CF.copy()
#     #find different features between original instance and counterfactual
#     nonoverlapping_features = np.where(orig_instance != CF, 1, 0) 
#     #nonoverlapping_features = np.where(orig_instance != CF)[1]
#     categorical_values = feature_ranges['categorical']
#     numerical_min_values = feature_ranges['numerical']['min']
#     numerical_max_values = feature_ranges['numerical']['max']

    
#     #print(f"Numerical Feature Index {idx}: Min = {min_val}, Max = {max_val}")
#     #find if different features are categorical or numerical
#     #apply related local_dp on each feature value
#     for i in range(len(nonoverlapping_features[0])):
#         if nonoverlapping_features[0,i] == 1:
#             x = CF[0,i]#the value of the feature in CF
#             #if i is in categoricals: 
#             selected_values = []
#             if(i in cat_features):
            
#                 for idx, values in categorical_values.items():
#                     if idx == i:
#                         feat_vals = values
#                         m,c = find_optimum_C_and_m(epsilon,len(feat_vals))
#                         for i in range(100):
#                             new_feature_value = Get_SRR_cat_feature(x,feat_vals,m,c,alpha_min)           # pass to the function the possible values of categorical feature
#                             selected_values.append(new_feature_value)
#                         break
#                 #find the list of possible values of the feature
#                 # if(lsrr == False):
#                 #     new_feature_value = get_ldp_cat_feature(x,feat_vals,epsilon,cat_type)           # pass to the function the possible values of categorical feature
#                 # else:
                    
            
#             #if i is numerical:
#             else:
#                 for idx, (min_val, max_val) in enumerate(zip(numerical_min_values, numerical_max_values)):
#                     if(num_featuers[idx] == i): 
#                         #find the min and max value of the feature
#                         m,c = find_optimum_C_and_m(epsilon,max_val - min_val)
                        
#                         for r in range(100):
#                             new_feature_value = Get_SRR_num_feature(x,min_val,max_val,m,c,alpha_min)        # pass to the function max and min values of the numerical_feature
#                             selected_values.append(new_feature_value)
                            
#                         break
#             num_bins = m
#             plt.hist(selected_values, bins=num_bins, edgecolor='k')
#             plt.xlabel('Value Range')
#             plt.ylabel('Frequency')
#             plt.title('Distribution of Numbers')
#             plt.show()
#             # ldp_CF[0,i] = new_feature_value
#         #else:
#             #do not change 
#             # the value in ldp-CF becuase it has not been changed
      
    
#     # return ldp_CF


