from abc import ABC,abstractmethod
from utils.optimization.reward import RewardFunction
from dpmechanisms.local_dp import add_ldp_noise
import numpy as np
import random
import copy

class optimization(ABC):
    @abstractmethod
    def optimize(self):
        pass

class best_first(optimization):
    def __init__(self,data,reward_function:RewardFunction):
        self.reward_function = reward_function
        self.data = data

    def optimize(self,NN):
        CF_candidate = self.data.X.copy()
        stop = False
        while stop == False:
            diff = np.where(CF_candidate != NN)[1]
            X_prune = np.tile(CF_candidate, (len(diff), 1))
            for r, c in enumerate(diff):
                X_prune[r, c] = NN[0, c]
            CF_candidate = self.reward_function.calculate_reward(X_prune,CF_candidate)
            if self.data.predict_fn(CF_candidate).argmax() in self.data.target_class:
                return CF_candidate
            
    def plaus_optimize(self,NN,changhable_features = None):
        CF_candidate = self.data.X.copy()
        stop = False
        while stop == False:
            ####ADDED TO APPLY ACTIONABLE FEATURES######
            orig_diff = np.where(CF_candidate != NN)[1]
            #just consider actionable features
#            diff = self.plausible_choices(orig_diff,CF_candidate)
            if(changhable_features != None):
                diff = [x for x in orig_diff if x in changhable_features]
            else:
                diff = orig_diff
            ####END ADDED TO APPLY ACTIONABLE FEATURES######
            if(len(diff) > 0):
                X_prune = np.tile(CF_candidate, (len(diff), 1))
                for r, c in enumerate(diff):
                    X_prune[r, c] = NN[0, c]
                CF_candidate = self.reward_function.calculate_reward(X_prune,CF_candidate)
                if self.data.predict_fn(CF_candidate).argmax() in self.data.target_class:
                    return CF_candidate
            else:

                # print("NO PLAUSIBLE NICE CF POSSIBLE")
                stop = True
        return NN
                
    
    
    def our_optimize(self,NN,k=1):
        # START PROCESS WITH THE INPUT INSTANCE, AND CHANGE ITS FEATURES DURING THE ALGORITHM
        CF_candidate = self.data.X.copy()
        stop = False
        counter = 0
        # SELECT ONE OF THE k NEIGHBOURS RANDOMLY AS BASIC INSTANCE TO GENERATE COUNTERFACTUAL
        random_candidate = np.array(random.choice(NN))
        basis_CF = random_candidate.reshape(1,-1)
        # KEEP THE PREDICTION OF BASIC INSTANCE TO COMPARE FOR IMPROVEMENT
        CF_prediction = self.data.predict_fn(CF_candidate)
        CF_prediction_class = CF_prediction.argmax()
        # GIVE AN INITIAL SCORE TO MAKE SURE WE IMPROVE 
        CF_score = -1000
        while stop == False:
            # NONOVERLAPPING FEATURES BETWEEN CF_CANDIDATE AND BASIC_CF ARE FOUND. AT THE BEGINING, CF_CANDIDATE IS EQUEALL TO INSTANCE X
            diff = np.where(CF_candidate != basis_CF)[1]
            # GENERATE A LIST OF ALL POSSIBLE CHANGES, FOR EACH NONOVERLAPPING FEATURE, K INSTANCES ARE INSERTED INTO X_PRUNE THAT GET THE VALUE OF THAT FEATURE FROM ONE OF THE K_NEIGHBOURTS SELECTED USING eXPONENTIAL PRIVACY
            X_prune = np.tile(CF_candidate, (len(diff)*k, 1))    
            i = 0
            for candidate in NN:      #in our case NN is a list of k neighbors selected using Exponential Mechanism
                for r, c in enumerate(diff):
                    X_prune[r+i, c] = candidate[c]  
                i+=len(diff)      #we add all neighbours here and in reward function we should add some randomness (probabely, not for sure)
            # NOW THAT ALL THE OPTIONS ARE SELECTED, WE SAVE EXISTING CANDIDATE THEN UPDATE NEXT FEATURE TO COMPARE THEIR PREDICTIONS, THIS WAY WE WONT DECREASE THE QUALITY OF SOLUTION BY CHOOSING ONE VALUE
            old_CF_candidate = CF_candidate
            CF_candidate = self.reward_function.calculate_reward(X_prune,CF_candidate)
            ######## added to prevent from getting into a loop between two options
            old_CF_candidate = CF_candidate
            old_CF_prediction = CF_prediction
            # IF NON OF THE EXISTING OPTIONS (FEATURE VALUSE CHOSEN FROM k NEIGHBOURS FOR REMAINING NON OVERLAPPING FEATURE) CANNOT IMPROVE THE RESULT, THEN NO COUNTERFACTUAL IS FOUND, 
            # EXCEPT THAT WE CHOOSE ANOTHER BASIC COUNTERFACTUAL  --------- CONSIDER TRYING THIS
            if(X_prune.size == 0):
                            # print("no plausible solution found")
                            #return np.array(basis_CF),np.array(basis_CF)
                            return basis_CF,basis_CF 
            CF_candidate = self.reward_function.calculate_reward(X_prune,CF_candidate)
            
            #     CF_candidate = old_CF_candidate
            #### to avoid from getting stalked in an infinity loop of not finding a cf
            if(np.array_equal(old_CF_candidate , CF_candidate)):
                counter +=1
            else:
                counter = 0


            ######## added to prevent from getting into a loop between two options
            # check it improves the result, replace it, continue
            # if it does not improve the result, remove it from x_prune, continue untile finding a good solution
            old_CF_prediction = CF_prediction
            old_CF_prediction_class = CF_prediction_class
            CF_prediction = self.data.predict_fn(CF_candidate)
            CF_prediction_class = CF_prediction.argmax()
            improved = False
            while(improved == False):   # here we search for better solution until solution improves or there is no other choice, which means no counterfactual exists
                if(CF_prediction_class != old_CF_prediction_class  or (CF_prediction_class == old_CF_prediction_class and CF_prediction[0,CF_prediction_class] < old_CF_prediction[0,CF_prediction_class])):   # if response improves the answer: means the prediction value of the current class for new CF is less than that of the old CF
                    improved = True
                    # we found a candidate that improves prediction
                else:  # the found candidate does not improve the solution, then remove it and search between remaining candidates
                    if(X_prune.size > 0): #we have some candidates yet
                        
                        X_prune = self.remove_candidate(X_prune,CF_candidate)
                        if(X_prune.size == 0):
                            # print("no DP solution found")
                            #return np.array(basis_CF),np.array(basis_CF)

                            return basis_CF,basis_CF 
                        CF_candidate = self.reward_function.calculate_reward(X_prune,old_CF_candidate)
                        CF_prediction = self.data.predict_fn(CF_candidate)
                        CF_prediction_class = CF_prediction.argmax()
            #### FirsT: compareS if the selection improves the solution or not
            #### Then, if it does not improve we need to do two things for next iteration:
            #  first, remove it from our choices until CF changes, 
            # Second, return it to the choices after changing old_Cf







            if self.data.predict_fn(CF_candidate).argmax() in self.data.target_class: # and CF_candidate[0] != basis_CF:
                #print("Basic instance:",basis_CF)
                return CF_candidate,basis_CF #np.array(basis_CF)
            #else:
            #    CF_candidate = self.data.X.copy()
            if counter >= (len(diff)):
                # print("no private solution found")
                #return np.array(basis_CF),np.array(basis_CF)
                return basis_CF,basis_CF
            


    def our_optimize_plausible(self,NN,changhable_features=None,increase_only=None,k=1):   # THE ONLY DIFFERENCE HERE WITH OUR_OPTIMIZATION IS THAT INSTEAD OF CHANGE COST, HERE WE USED CHANGABLE FEATURES, WHICH MEANS WE FORBID SOME FEATURES LIKE RACE AND SEX TO BE SUGGESTED TO CHANGE IN OUR COUNTERFACTUAL
        # START PROCESS WITH THE INPUT INSTANCE, AND CHANGE ITS FEATURES DURING THE ALGORITHM
        CF_candidate = self.data.X.copy()
        
        # KEEP THE PREDICTION OF BASIC INSTANCE TO COMPARE FOR IMPROVEMENT
        CF_prediction = self.data.predict_fn(CF_candidate)
        CF_prediction_class = CF_prediction.argmax()
        found = False
        # until no solution is found and yet we have other initial points to start with, choose basic instance
        # PROBLEM: wen removing previous choices, our privacy leakage at the las iteration increase, need to choose from remaining neighbours, but keep the others to generaye CF
        while (NN != None and found == False) and len(NN) >0:
            # GIVE AN INITIAL SCORE TO MAKE SURE WE IMPROVE     
            # CF_score = -1000
            stop = False
            counter = 0
           # SELECT ONE OF THE k NEIGHBOURS RANDOMLY AS BASIC INSTANCE TO GENERATE COUNTERFACTUAL            
           # choose randomely one array from NN and remove it from the NN at the same time
            basis_CF = NN.pop(random.choice(range(len(NN)))).reshape(1,-1)
            
            while stop == False:
                # NONOVERLAPPING FEATURES BETWEEN CF_CANDIDATE AND BASIC_CF ARE FOUND. AT THE BEGINING, CF_CANDIDATE IS EQUEALL TO INSTANCE X
                orig_diff = np.where(CF_candidate != basis_CF)[1]
                #just consider actionable features
    #            diff = self.plausible_choices(orig_diff,CF_candidate)
                if(changhable_features != None):
                    diff = [x for x in orig_diff if x in changhable_features]
                else:
                    diff = orig_diff
                # GENERATE A LIST OF ALL POSSIBLE CHANGES, FOR EACH NONOVERLAPPING FEATURE, K INSTANCES ARE INSERTED INTO X_PRUNE THAT GET THE VALUE OF THAT FEATURE FROM ONE OF THE K_NEIGHBOURTS SELECTED USING eXPONENTIAL PRIVACY
                X_prune = np.tile(CF_candidate, (len(diff)*k, 1))    #generate all possible changes for all features
                i = 0
                for candidate in NN:      #in our case NN is a list of k neighbors selected using Exponential Mechanism
                    for r, c in enumerate(diff):
                        X_prune[r+i, c] = candidate[c]  
                    i+=len(diff)      #we add all neighbours here and in reward function we should add some randomness (probabely, not for sure)
                # NOW THAT ALL THE OPTIONS ARE SELECTED, WE SAVE EXISTING CANDIDATE THEN UPDATE NEXT FEATURE TO COMPARE THEIR PREDICTIONS, THIS WAY WE WONT DECREASE THE QUALITY OF SOLUTION BY CHOOSING ONE VALUE
                old_CF_candidate = CF_candidate
                old_CF_prediction = CF_prediction
                # IF NON OF THE EXISTING OPTIONS (FEATURE VALUSE CHOSEN FROM k NEIGHBOURS FOR REMAINING NON OVERLAPPING FEATURE) CANNOT IMPROVE THE RESULT, THEN NO COUNTERFACTUAL IS FOUND, 
                # EXCEPT THAT WE CHOOSE ANOTHER BASIC COUNTERFACTUAL  --------- CONSIDER TRYING THIS
                if(X_prune.size == 0):
                                #print("no plausible solution found")
                                #return np.array(basis_CF),np.array(basis_CF)
                                #return basis_CF,basis_CF 
                                stop = True
                else:
                    CF_candidate = self.reward_function.calculate_reward(X_prune,CF_candidate)
                        
                # old_CF_score = CF_score
                # CF_candidate,CF_score = self.reward_function.calculate_reward(X_prune,CF_candidate)
                # if(CF_score <= old_CF_score):
                #     CF_score = old_CF_score
                #     CF_candidate = old_CF_candidate
                #### to avoid from getting stalked in an infinity loop of not finding a cf
                            
                if(np.array_equal(old_CF_candidate , CF_candidate)):
                    counter +=1
                
                else:
                    counter = 0
                
                
                # check it improves the result, replace it, continue
                # if it does not improve the result, remove it from x_prune, continue untile finding a good solution
                old_CF_prediction = CF_prediction
                old_CF_prediction_class = CF_prediction_class
                CF_prediction = self.data.predict_fn(CF_candidate)
                CF_prediction_class = CF_prediction.argmax()
                improved = False
                failed_search = False
                while(improved == False and failed_search == False):   # here we search for better solution until solution improves or there is no other choice, which means no counterfactual exists
                    ######## added to prevent from getting into a loop between two options
                    #### FirsT: compareS if the selection improves the solution or not
                   
                    if(CF_prediction_class != old_CF_prediction_class  or (CF_prediction_class == old_CF_prediction_class and CF_prediction[0,CF_prediction_class] < old_CF_prediction[0,CF_prediction_class])):   # if response improves the answer: means the prediction value of the current class for new CF is less than that of the old CF
                        improved = True
                        # we found a candidate that improves prediction
                    else:  # the found candidate does not improve the solution, then remove it and search between remaining candidates
                         #### Then, if it does not improve we need to do two things for next iteration:
                        #  first, remove it from our choices until CF changes, 
                        # Second, return it to the choices after changing old_Cf
                        if(X_prune.size > 0): #we have some candidates yet
                            
                            X_prune = self.remove_candidate(X_prune,CF_candidate)
                            if(X_prune.size == 0):
                                #print("no plausible solution found")
                                #return np.array(basis_CF),np.array(basis_CF)
                                #return basis_CF,basis_CF 
                                stop = True
                                failed_search = True
                            else:
                                CF_candidate = self.reward_function.calculate_reward(X_prune,old_CF_candidate)
                                CF_prediction = self.data.predict_fn(CF_candidate)
                                CF_prediction_class = CF_prediction.argmax()
                        else:
                            #THERE IS NO OTHER CHOICE TO CHANGE A FEATURE VALUE AND IMPROVE THE SOLUTION
                            failed_search = True
                
                #### FirsT: compareS if the selection improves the solution or not
                #### Then, if it does not improve we need to do two things for next iteration:
                #  first, remove it from our choices until CF changes, 
                # Second, return it to the choices after changing old_Cf
                
                if CF_prediction_class in self.data.target_class: # and CF_candidate[0] != basis_CF:
                #    print("Basic instance:",basis_CF)
                    return CF_candidate,basis_CF #np.array(basis_CF)
                #else:
                #    CF_candidate = self.data.X.copy()
                if counter >= (len(diff)):
                    #print("no plausible solution found")
                    #return np.array(basis_CF),np.array(basis_CF)
                    #return basis_CF,basis_CF
                    stop = True
        #HERE WE SHOULD RETURN ANOTHER SOLUTION NOT TO LEAK PRIVACY
        # print("No Result found from This Neighbor list")
        
        return basis_CF,basis_CF
            
    def remove_candidate(self,candidate_list,bad_candidate): # remove candidate that results in an infinite loop between too candidates
        #new_x_prune = np.array(candidate_list.shape) #np.array() # np.array(candidate_list.shape())
        #new_x_prune[:] = [item for item in candidate_list if not np.all(item == bad_candidate)] # [item for item in candidate_list if item != bad_candidate]
        new_x_prune = candidate_list[~(candidate_list == bad_candidate).all(1)]
        return new_x_prune

    def ChooseBasicCF(self,neighborlist):
        selectedindex = random.choice(range(len(neighborlist)))
        SelectedNeighbor = neighborlist[selectedindex] # random.choice(neighborlist)
        newneighborlist = [arr for arr in neighborlist if not np.array_equal(arr, SelectedNeighbor)]
        #neighborlist.remove(neighborlist, selectedindex, 0)
        basic = SelectedNeighbor.reshape(1,-1)
        #neighborlist = neighborlist[~(neighborlist == SelectedNeighbor).all(1)]
        return basic,newneighborlist

    
    def ldp_optimize(self,NN,feature_ranges,epsilon,changhable_features=None,increase_only=None,k=1,ldp_mechanism = 0):   # THE ONLY DIFFERENCE HERE WITH OUR_OPTIMIZATION IS THAT INSTEAD OF CHANGE COST, HERE WE USED CHANGABLE FEATURES, WHICH MEANS WE FORBID SOME FEATURES LIKE RACE AND SEX TO BE SUGGESTED TO CHANGE IN OUR COUNTERFACTUAL
        # START PROCESS WITH THE INPUT INSTANCE, AND CHANGE ITS FEATURES DURING THE ALGORITHM
        CF_candidate = copy.deepcopy(self.data.X)
        
        # KEEP THE PREDICTION OF BASIC INSTANCE TO COMPARE FOR IMPROVEMENT
        CF_prediction = self.data.predict_fn(CF_candidate)
        CF_prediction_class = CF_prediction.argmax()
        found = False
        
        
        # until no solution is found and yet we have other initial points to start with, choose basic instance
        # PROBLEM: wen removing previous choices, our privacy leakage at the las iteration increase, need to choose from remaining neighbours, but keep the others to generaye CF
        while (NN != None and found == False) and len(NN) >0:
            # GIVE AN INITIAL SCORE TO MAKE SURE WE IMPROVE     
            # CF_score = -1000l
            stop = False
            counter = 0
           # SELECT ONE OF THE k NEIGHBOURS RANDOMLY AS BASIC INSTANCE TO GENERATE COUNTERFACTUAL            
           # choose randomely one array from NN and remove it from the NN at the same time
            ldp_NN = copy.deepcopy(NN)
            ## find the index of basic_CF to be able to replace it with its ldp version for computations
            randind = random.choice(range(len(NN)))
            basis_CF = NN.pop(randind).reshape(1,-1)
            # print("basic_CF:",basis_CF)
                # NONOVERLAPPING FEATURES BETWEEN CF_CANDIDATE AND BASIC_CF ARE FOUND. AT THE BEGINING, CF_CANDIDATE IS EQUEALL TO INSTANCE X
            orig_diff = np.where(CF_candidate != basis_CF)[1]
                #just consider actionable features
    #            diff = self.plausible_choices(orig_diff,CF_candidate)
            if(changhable_features != None):
                diff = [x for x in orig_diff if x in changhable_features]
            else:
                diff = orig_diff
            # GENERATE A LIST OF ALL POSSIBLE CHANGES, FOR EACH NONOVERLAPPING FEATURE, K INSTANCES ARE INSERTED INTO X_PRUNE THAT GET THE VALUE OF THAT FEATURE FROM ONE OF THE K_NEIGHBOURTS SELECTED USING eXPONENTIAL PRIVACY
            # print("before noise:",ldp_NN)
            ldp_NN = add_ldp_noise(ldp_NN,diff,feature_ranges,epsilon,ldp_mechanism)                
            # print("after noise:",ldp_NN)
            basis_CF = ldp_NN.pop(randind).reshape(1,-1)
            # print("basic_cf_after noise:",basis_CF)
            
            
            while stop == False:
                # NONOVERLAPPING FEATURES BETWEEN CF_CANDIDATE AND BASIC_CF ARE FOUND. AT THE BEGINING, CF_CANDIDATE IS EQUEALL TO INSTANCE X
                orig_diff = np.where(CF_candidate != basis_CF)[1]
                #just consider actionable features
    #            diff = self.plausible_choices(orig_diff,CF_candidate)
                if(changhable_features != None):
                    diff = [x for x in orig_diff if x in changhable_features]
                else:
                    diff = orig_diff
                    X_prune = np.tile(CF_candidate, (len(diff)*k, 1))    #generate all possible changes for all features
                    i = 0
                ####---------------------- here, instead of NN we generate a list of candidates which their feat_values on
                    #############################################
                    #here we should add noice to nonoverlapping featurevalues in all neighbours
                    #1. find nonoverlapping features   
                    #2. add noise to all nonoverlapping features in NN
                
                    #3. Generate CF 
                    #4. if added new instance we should update epsilon and everything for noise generation, so we need original values of instances
                    ##############################################    
                ##########----------------
                for candidate in ldp_NN:      #in our case NN is a list of k neighbors selected using Exponential Mechanism
                    for r, c in enumerate(diff):
                        X_prune[r+i, c] = candidate[0,c]  
                    i+=len(diff)      #we add all neighbours here and in reward function we should add some randomness (probabely, not for sure)
                # NOW THAT ALL THE OPTIONS ARE SELECTED, WE SAVE EXISTING CANDIDATE THEN UPDATE NEXT FEATURE TO COMPARE THEIR PREDICTIONS, THIS WAY WE WONT DECREASE THE QUALITY OF SOLUTION BY CHOOSING ONE VALUE
                old_CF_candidate = CF_candidate
                old_CF_prediction = CF_prediction
                # IF NON OF THE EXISTING OPTIONS (FEATURE VALUSE CHOSEN FROM k NEIGHBOURS FOR REMAINING NON OVERLAPPING FEATURE) CANNOT IMPROVE THE RESULT, THEN NO COUNTERFACTUAL IS FOUND, 
                # EXCEPT THAT WE CHOOSE ANOTHER BASIC COUNTERFACTUAL  --------- CONSIDER TRYING THIS
                if(X_prune.size == 0):
                                #print("no plausible solution found")
                                #return np.array(basis_CF),np.array(basis_CF)
                                #return basis_CF,basis_CF 
                                stop = True
                else:
                    CF_candidate = self.reward_function.calculate_reward(X_prune,CF_candidate)
                        
                # old_CF_score = CF_score
                # CF_candidate,CF_score = self.reward_function.calculate_reward(X_prune,CF_candidate)
                # if(CF_score <= old_CF_score):
                #     CF_score = old_CF_score
                #     CF_candidate = old_CF_candidate
                #### to avoid from getting stalked in an infinity loop of not finding a cf
                            
                if(np.array_equal(old_CF_candidate , CF_candidate)):
                    counter +=1
                
                else:
                    counter = 0
                
                
                # check it improves the result, replace it, continue
                # if it does not improve the result, remove it from x_prune, continue untile finding a good solution
                old_CF_prediction = CF_prediction
                old_CF_prediction_class = CF_prediction_class
                CF_prediction = self.data.predict_fn(CF_candidate)
                CF_prediction_class = CF_prediction.argmax()
                improved = False
                failed_search = False
                while(improved == False and failed_search == False):   # here we search for better solution until solution improves or there is no other choice, which means no counterfactual exists
                    ######## added to prevent from getting into a loop between two options
                    #### FirsT: compareS if the selection improves the solution or not
                   
                    if(CF_prediction_class != old_CF_prediction_class  or (CF_prediction_class == old_CF_prediction_class and CF_prediction[0,CF_prediction_class] < old_CF_prediction[0,CF_prediction_class])):   # if response improves the answer: means the prediction value of the current class for new CF is less than that of the old CF
                        improved = True
                        # we found a candidate that improves prediction
                    else:  # the found candidate does not improve the solution, then remove it and search between remaining candidates
                         #### Then, if it does not improve we need to do two things for next iteration:
                        #  first, remove it from our choices until CF changes, 
                        # Second, return it to the choices after changing old_Cf
                        if(X_prune.size > 0): #we have some candidates yet
                            
                            X_prune = self.remove_candidate(X_prune,CF_candidate)
                            if(X_prune.size == 0):
                                #print("no plausible solution found")
                                #return np.array(basis_CF),np.array(basis_CF)
                                #return basis_CF,basis_CF 
                                stop = True
                                failed_search = True
                            else:
                                CF_candidate = self.reward_function.calculate_reward(X_prune,old_CF_candidate)
                                CF_prediction = self.data.predict_fn(CF_candidate)
                                CF_prediction_class = CF_prediction.argmax()
                        else:
                            #THERE IS NO OTHER CHOICE TO CHANGE A FEATURE VALUE AND IMPROVE THE SOLUTION
                            failed_search = True
                
                #### FirsT: compareS if the selection improves the solution or not
                #### Then, if it does not improve we need to do two things for next iteration:
                #  first, remove it from our choices until CF changes, 
                # Second, return it to the choices after changing old_Cf
                
                if CF_prediction_class in self.data.target_class: # and CF_candidate[0] != basis_CF:
                #    print("Basic instance:",basis_CF)
                    return CF_candidate,basis_CF #np.array(basis_CF)
                #else:
                #    CF_candidate = self.data.X.copy()
                if counter >= (len(diff)):
                    #print("no plausible solution found")
                    #return np.array(basis_CF),np.array(basis_CF)
                    #return basis_CF,basis_CF
                    stop = True
        #HERE WE SHOULD RETURN ANOTHER SOLUTION NOT TO LEAK PRIVACY
        # print("No Result found from This Neighbor list")
        
        return basis_CF,basis_CF