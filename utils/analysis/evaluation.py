import numpy as np
def reidentify(instance,dataset):
    count = 0
    for i in range(len(dataset)):
        a = np.where(instance == dataset[i])[1]
        if len(a) == len(instance[0]) :
            count +=1

    return count

def compare_solutions(Factual_instance,basic,CF):   #find how the CF changes in comparison with basic_cf and original instance
        eps = 0.00000000001
        # differencses between Factual_instance and basic Counterfacual
        nonoverlapping_features = np.where(Factual_instance != basic)[1]
        nonoverlapping = len(nonoverlapping_features)   # np.count_nonzero(nonoverlapping_features)
        #difference between Factual_instance and generated counterfactual
        Changes_features = np.where(Factual_instance != CF)[1]
        Changed = len(Changes_features) # np.count_nonzero(Changes_features)
        # ratio of features selected from basic_CF   : in changed feutures, find what is equal to basic
        
        Chosen_from_basic_features = np.where(np.logical_and(Factual_instance != CF , CF == basic))[1]
        Chosen_from_basic = len(Chosen_from_basic_features) # np.count_nonzero(Chosen_from_basic_features)
        # ratio of changes features
        changed_ratio = Changed / len(Factual_instance[0]) # ( nonoverlapping + eps )  #to avoid devision by zero
        # ration of selected from basic instance
        from_basic_ratio = Chosen_from_basic / ( Changed + eps)   #to avoid devision by zero
        return changed_ratio,from_basic_ratio

