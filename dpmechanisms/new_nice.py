#from utils.distance import*
import sys
import copy
#sys.path.append('F:\PhD\pythonprojects\dpnice')
sys.path.append('dpnice')

from utils.data.distance import *
#from nice.utils.data import data_NICE
from utils.data.data import data_NICE
from utils.optimization.heuristic import best_first
#from nice.utils.optimization.heuristic import best_first
from utils.optimization.reward import SparsityReward, ProximityReward, PlausibilityReward,DifferenriallPrivacyAward
#from nice.utils.optimization.reward import SparsityReward, ProximityReward, PlausibilityReward,DifferenriallPrivacyAward
from typing import Optional
import numpy as np

# =============================================================================
# Types and constants
# =============================================================================
CRITERIA_DIS = {'HEOM':HEOM}
CRITERIA_NRM = {'std':StandardDistance,
                'minmax':MinMaxDistance}
CRITERIA_REW = {'sparsity':SparsityReward,
                'proximity':ProximityReward,
                'plausibility':PlausibilityReward,
                'differentialprivacy':DifferenriallPrivacyAward}


class NICE:
    def __init__(
            self,
            predict_fn,
            X_train:np.ndarray,
            cat_feat:list,
            sensitivity:float = 1.0,
            num_feat ='auto',
            y_train: Optional[np.ndarray]=None,
            optimization='sparsity',
            justified_cf:bool = True,
            distance_metric:str ='HEOM',
            num_normalization:str = 'minmax',
            auto_encoder = None):

        self.optimization = optimization
        self.data = data_NICE(X_train,y_train,cat_feat,num_feat,predict_fn,justified_cf,0.00000000001)
        self.distance_metric = CRITERIA_DIS[distance_metric](self.data, CRITERIA_NRM[num_normalization])
        self.nearest_neighbour = NearestNeighbour(self.data, self.distance_metric)
        #added by me to generate tha farthest instace for not found CF
        self.NotFoundCF = [max(elements) for elements in zip(*self.data.X_train)]
        self.NICENotFoundCF = 0
        self.DPNotfoundCF = 0
        #added by me 
        self.PrivateNearestNeighbour = PrivateNearestNeighbour(self.data, self.distance_metric)
        self.sensitivity = self.distance_metric.max_distance()  #sensitivity(X_train)
        if optimization != 'none':
            self.reward_function = CRITERIA_REW[optimization](
                self.data,
                distance_metric = self.distance_metric,
                auto_encoder= auto_encoder
            )
            self.optimizer = best_first(self.data,self.reward_function)


    

    def explain(self,X,target_class ='other'):#todo target class 'other'
        self.data.fit_to_X(X,target_class)
        NN = self.nearest_neighbour.find_neighbour(self.data.X)
        if self.optimization != 'none':
            CF = self.optimizer.optimize(NN)
            return CF
        return NN
    

    def plaus_explain(self,X,target_class ='other',changhable_features=None):#todo target class 'other'
        self.data.fit_to_X(X,target_class)
        NN = self.nearest_neighbour.find_plause_neighbour(self.data.X,changhable_features)# find_neighbour(self.data.X)
        if self.optimization != 'none':
            if NN is not None:
                CF = self.optimizer.plaus_optimize(NN,changhable_features)
                return CF,0   #it could find CF, so n_tries = 0
            else:
                # count the number of not found solutions, this case should never happen becuase we always have options!
                self.NICENotFoundCF +=1
                # print("nice was unable to find a plausible solution")
                return NN,self.NotFoundCF   #return the basic instance, which is not plausible
        return NN,0

    def explain_Second(self,X,target_class ='other'):#todo target class 'other'
        self.data.fit_to_X(X,target_class)
        NN = self.nearest_neighbour.find_neighbour(self.data.X)
        if self.optimization != 'none':
            CF = self.optimizer.optimize(NN)
            return CF,NN
        return NN,NN
    
    def synth_explain(self,X,target_class ='other',DS = None):
        if(DS is None):
            print("No dataset assign to find CF from,runnong regular NICE")
            return self.explain_Second(self,X,target_class)
        else: 
            candidate_list = self.data.fit_DS_to_X(X,target_class,DS) #fit_to_X(X,target_class)
            NN = self.nearest_neighbour.find_neighbour_from_candidates(self.data.X,candidate_list) 
            if self.optimization != 'none':
                CF = self.optimizer.optimize(NN)
                return CF,NN,0
            return NN,NN,0


    #we should not use this function anymore
    def private_explain(self,X,target_class ='other',change_cost=None,K=1,epsilon = 1):#todo target class 'other'
        self.data.fit_to_X(X,target_class)
        if change_cost is None:
            change_cost= np.ones_like(X)
        NN = self.PrivateNearestNeighbour.find_K_neighbours(change_cost,self.data.X,K,epsilon)
        if self.optimization != 'none':
            CF,basic_CF = self.optimizer.our_optimize(NN,K) #    optimize(NN)
            return CF,basic_CF
        return NN
    

    def private_explain_plausible(self,X,target_class ='other',change_cost=None,changhable_features=None,increase_only=None,K=1,epsilon = 1,sensitivity =1):#todo target class 'other'
        epsilon = epsilon / K + 0.00000000001
        self.data.fit_to_X(X,target_class)
        if change_cost is None:
            change_cost= np.ones_like(X)
        NoDPCFFound = True
        NN = self.PrivateNearestNeighbour.find_K_neighbours(change_cost,self.data.X,K,epsilon,sensitivity)
        orig_NN = copy.deepcopy(NN) # NN.copy()
        # to save  how many trys it took to acheive a counterfactual for an instance
        n_try = 0
        while(NoDPCFFound == True):
            if self.optimization != 'none':
                #CF,basic_CF = self.optimizer.our_optimize(NN,changhable_features,increase_only,K) #    optimize(NN)
                CF,basic_CF = self.optimizer.our_optimize_plausible(NN,changhable_features,increase_only,K) #    optimize(NN)
                # if CF != basic_CF it means the plausible solution found
                if(not np.array_equal(CF,basic_CF)):
                    return CF,basic_CF,n_try
                #if a plausible solution is not found, then we have returned basic_cf, this means we need to add another neighbour to improve the result
                else:
                    self.DPNotfoundCF += 1
                    additional_neighbour = self.PrivateNearestNeighbour.find_K_neighbours(change_cost,self.data.X,1,epsilon,sensitivity)
                    orig_NN.append(additional_neighbour[0])
                    NN = orig_NN.copy()
                    K+=1    #because in next call of our_optimize_plausible we will have 1 more neighbor in the list
                    n_try +=1   # increase the number of runs
            #return NN
    
    def private_explain_plausible_working_AAAI(self,X,target_class ='other',change_cost=None,changhable_features=None,increase_only=None,K=1,epsilon = 1):#todo target class 'other'
        epsilon = epsilon / K + 0.00000000001
        self.data.fit_to_X(X,target_class)
        if change_cost is None:
            change_cost= np.ones_like(X)
        NoDPCFFound = True
        NN = self.PrivateNearestNeighbour.find_K_neighbours(change_cost,self.data.X,K,epsilon)
        orig_NN = copy.deepcopy(NN) # NN.copy()
        # to save  how many trys it took to acheive a counterfactual for an instance
        n_try = 0
        while(NoDPCFFound == True):
            if self.optimization != 'none':
                #CF,basic_CF = self.optimizer.our_optimize(NN,changhable_features,increase_only,K) #    optimize(NN)
                CF,basic_CF = self.optimizer.our_optimize_plausible(NN,changhable_features,increase_only,K) #    optimize(NN)
                # if CF != basic_CF it means the plausible solution found
                if(not np.array_equal(CF,basic_CF)):
                    return CF,basic_CF,n_try
                #if a plausible solution is not found, then we have returned basic_cf, this means we need to add another neighbour to improve the result
                else:
                    self.DPNotfoundCF += 1
                    additional_neighbour = self.PrivateNearestNeighbour.find_K_neighbours(change_cost,self.data.X,1,epsilon)
                    orig_NN.append(additional_neighbour[0])
                    NN = orig_NN.copy()
                    K+=1    #because in next call of our_optimize_plausible we will have 1 more neighbor in the list
                    n_try +=1   # increase the number of runs
            #return NN

    def private_explain_plausible_fail_stop(self,X,target_class ='other',change_cost=None,changhable_features=None,increase_only=None,K=1,epsilon = 1 , sensitivity = 1):#todo target class 'other'
        epsilon = epsilon / K + 0.00000000001
        self.data.fit_to_X(X,target_class)
        if change_cost is None:
            change_cost= np.ones_like(X)
        NoDPCFFound = True
        NN = self.PrivateNearestNeighbour.find_K_neighbours(change_cost,self.data.X,K,epsilon,sensitivity=sensitivity)
        orig_NN = copy.deepcopy(NN) # NN.copy()
        # to save  how many trys it took to acheive a counterfactual for an instance
        n_try = 0
        while(NoDPCFFound == True):
            if self.optimization != 'none':
                #CF,basic_CF = self.optimizer.our_optimize(NN,changhable_features,increase_only,K) #    optimize(NN)
                CF,basic_CF = self.optimizer.our_optimize_plausible(NN,changhable_features,increase_only,K) #    optimize(NN)
                # if CF != basic_CF it means the plausible solution found
                if(not np.array_equal(CF,basic_CF)):
                    return CF,basic_CF,n_try
                # else means the first randomization failed to find CF
                else:
                    return basic_CF,basic_CF,1
                # #if a plausible solution is not found, then we have returned basic_cf, this means we need to add another neighbour to improve the result
                # else:
                #     self.DPNotfoundCF += 1
                #     additional_neighbour = self.PrivateNearestNeighbour.find_K_neighbours(change_cost,self.data.X,1,epsilon)
                #     orig_NN.append(additional_neighbour[0])
                #     NN = orig_NN.copy()
                #     K+=1    #because in next call of our_optimize_plausible we will have 1 more neighbor in the list
                #     n_try +=1   # increase the number of runs
            #return NN

    def private_explain_plausible_fail_stop_working_AAAI(self,X,target_class ='other',change_cost=None,changhable_features=None,increase_only=None,K=1,epsilon = 1):#todo target class 'other'
        epsilon = epsilon / K + 0.00000000001
        self.data.fit_to_X(X,target_class)
        if change_cost is None:
            change_cost= np.ones_like(X)
        NoDPCFFound = True
        NN = self.PrivateNearestNeighbour.find_K_neighbours(change_cost,self.data.X,K,epsilon)
        orig_NN = copy.deepcopy(NN) # NN.copy()
        # to save  how many trys it took to acheive a counterfactual for an instance
        n_try = 0
        while(NoDPCFFound == True):
            if self.optimization != 'none':
                #CF,basic_CF = self.optimizer.our_optimize(NN,changhable_features,increase_only,K) #    optimize(NN)
                CF,basic_CF = self.optimizer.our_optimize_plausible(NN,changhable_features,increase_only,K) #    optimize(NN)
                # if CF != basic_CF it means the plausible solution found
                if(not np.array_equal(CF,basic_CF)):
                    return CF,basic_CF,n_try
                # else means the first randomization failed to find CF
                else:
                    return basic_CF,basic_CF,1
                # #if a plausible solution is not found, then we have returned basic_cf, this means we need to add another neighbour to improve the result
                # else:
                #     self.DPNotfoundCF += 1
                #     additional_neighbour = self.PrivateNearestNeighbour.find_K_neighbours(change_cost,self.data.X,1,epsilon)
                #     orig_NN.append(additional_neighbour[0])
                #     NN = orig_NN.copy()
                #     K+=1    #because in next call of our_optimize_plausible we will have 1 more neighbor in the list
                #     n_try +=1   # increase the number of runs
            #return NN



    def get_exp_results(self,X,target_class ='other',change_cost=None):

        self.data.fit_to_X(X,target_class)
        if change_cost is None:
            change_cost= np.ones_like(X)
        scores,distances,selection_count,selected_scores = self.PrivateNearestNeighbour.Test_exp_mech(change_cost,X)
        return scores,distances,selection_count,selected_scores
    


    def private_explain_LDP(self,X,feature_ranges,target_class ='other',change_cost=None,changhable_features=None,increase_only=None,K=1,epsilon = 1,ldp_mech = 0):#todo target class 'other'
        # epsilon = epsilon / K + 0.00000000001
        self.data.fit_to_X(X,target_class)
        if change_cost is None:
            change_cost= np.zeros_like(X)   ### here it is set to zero to make sure about its competency with nice
        NoDPCFFound = True
        NN = self.nearest_neighbour.find_K_neighbours(self.data.X,K) # PrivateNearestNeighbour.find_K_neighbours(change_cost,self.data.X,K,epsilon)
        orig_NN = copy.deepcopy(NN)
        print("orig_NN:",orig_NN)
        # to save  how many trys it took to acheive a counterfactual for an instance
        n_try = 0
        while(NoDPCFFound == True):
            print("orig_NN:",orig_NN)
            print("NN:",NN)
            if self.optimization != 'none':
                #CF,basic_CF = self.optimizer.our_optimize(NN,changhable_features,increase_only,K) #    optimize(NN)
                CF,basic_CF = self.optimizer.ldp_optimize(NN=NN,k=K,feature_ranges=feature_ranges,epsilon=epsilon,ldp_mechanism=ldp_mech) # our_optimize_plausible(NN,changhable_features,increase_only,K) #    optimize(NN)
                # if CF != basic_CF it means the plausible solution found
                if(not np.array_equal(CF,basic_CF)):
                    return CF,basic_CF,n_try
                #if a plausible solution is not found, then we have returned basic_cf, this means we need to add another neighbour to improve the result
                else:
                    self.DPNotfoundCF += 1
                    additional_neighbour = self.nearest_neighbour.find_K_neighbours(self.data.X,1) #PrivateNearestNeighbour.find_K_neighbours(change_cost,self.data.X,1,epsilon)
                    orig_NN.append(additional_neighbour[0])
                    NN = orig_NN.copy()
                    K+=1    #because in next call of our_optimize_plausible we will have 1 more neighbor in the list
                    n_try +=1   # increase the number of runs
            #return NN
    # apply laplacien noise on nonoverlaping features of all neighbours to implement inline_ldp
    def private_explain_LDP_early_stop(self,X,feature_ranges,target_class ='other',change_cost=None,changhable_features=None,increase_only=None,K=1,epsilon = 1,ldp_mech = 0):#todo target class 'other'
        # epsilon = epsilon / K + 0.00000000001
        self.data.fit_to_X(X,target_class)
        if change_cost is None:
            change_cost= np.zeros_like(X)   ### here it is set to zero to make sure about its competency with nice
        NoDPCFFound = True
        NN = self.nearest_neighbour.find_K_neighbours(self.data.X,K) # PrivateNearestNeighbour.find_K_neighbours(change_cost,self.data.X,K,epsilon)
        orig_NN = copy.deepcopy(NN)
        # print("orig_NN:",orig_NN)
        # to save  how many trys it took to acheive a counterfactual for an instance
        n_try = 0
        while(NoDPCFFound == True):
            # print("orig_NN:",orig_NN)
            # print("NN:",NN)
            if self.optimization != 'none':
                #CF,basic_CF = self.optimizer.our_optimize(NN,changhable_features,increase_only,K) #    optimize(NN)
                CF,basic_CF = self.optimizer.ldp_optimize(NN=NN,k=K,feature_ranges=feature_ranges,epsilon=epsilon,ldp_mechanism=ldp_mech) # our_optimize_plausible(NN,changhable_features,increase_only,K) #    optimize(NN)
                # if CF != basic_CF it means the plausible solution found
                if(not np.array_equal(CF,basic_CF)):
                    return CF,basic_CF,n_try
                else:  # if no plausible solution found, return the basic instance instead of CF. by setting n_try =1 we show that it is not CF
                    return basic_CF,basic_CF,1
                #if a plausible solution is not found, then we have returned basic_cf, this means we need to add another neighbour to improve the result
                # else:
                #     self.DPNotfoundCF += 1
                #     additional_neighbour = self.nearest_neighbour.find_K_neighbours(self.data.X,1) #PrivateNearestNeighbour.find_K_neighbours(change_cost,self.data.X,1,epsilon)
                #     orig_NN.append(additional_neighbour[0])
                #     NN = orig_NN.copy()
                #     K+=1    #because in next call of our_optimize_plausible we will have 1 more neighbor in the list
                #     n_try +=1   # increase the number of runs
            #return NN

    ### for CF methods like nice,exp_nice, inlinie ldp and ldp, DS is already fit to X, and Candidate view is already generated (the CF class has been specified)
    ### for methods that use synthetic data to generate CFs, we need to fit DS to X to find candidate_view, then select its subset to calculate X's distanse to make sure about plausibility
    def Calculate_Plaus_Dists(self,Generated_CF, k=10, should_fit = False, X = None , target_class = 'other'):
        plause_dists = []  # a list of 3 distances used for plausibility:
                           #  1. distance to nearest instance in CF class
                           #  2. Avg distance to KNN instances to CF class
                           #  3. Avg. distance to a random subset of instances in CF class
        inf = 100000
        
        if(should_fit):
            if(X is None ):
                print("No initial instance is provided")
                return inf
            else:
                #  1. distance to nearest instance in CF class
                self.data.fit_to_X(X,target_class)

        NN = self.nearest_neighbour.find_neighbour(Generated_CF)
        plause_dists.append(self.distance_metric.measure(Generated_CF,NN))
        #  2. Avg distance to KNN instances to CF class
        NN = self.nearest_neighbour.find_K_neighbours(Generated_CF,k)
        dist = 0.0
        for neighber in NN:
            dist += self.distance_metric.measure(Generated_CF,neighber)
        plause_dists.append(dist/k)
        #  3. Avg. distance to a random subset of instances in CF class
        NN = self.nearest_neighbour.find_K_neighbours(Generated_CF,k,random=True)
        dist = 0.0
        for neighber in NN:
            dist += self.distance_metric.measure(Generated_CF,neighber)
        plause_dists.append(dist/k)

        return plause_dists