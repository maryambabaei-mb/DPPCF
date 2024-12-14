from abc import ABC,abstractmethod
import numpy as np
import sys
sys.path.append('F:\PhD\pythonprojects\dpnice')
from utils.data.distance import DistanceMetric


class RewardFunction(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def calculate_reward(self):
        pass


class SparsityReward(RewardFunction):

    def __init__(self,data,**kwargs):
        self.data = data
        pass

    def calculate_reward(self,X_prune, previous_CF_candidate):
        score_prune = -self.data.predict_fn(previous_CF_candidate) + self.data.l(X_prune)
        score_diff = score_prune[:, self.data.target_class]# - score_prune[:, self.data.X_class][:,np.newaxis] #multiclas
        score_diff = score_diff.max(axis = 1)
        idx_max = np.argmax(score_diff)
        CF_candidate = X_prune[idx_max:idx_max+1, :]
        return CF_candidate


class ProximityReward(RewardFunction):
    def __init__(self,data,distance_metric:DistanceMetric,**kwargs):
        self.data = data
        self.distance_metric = distance_metric
    def calculate_reward(self,X_prune, previous_CF_candidate):
        score_prune = self.data.predict_fn(X_prune)
        score_diff = self.data.predict_fn(X_prune)[:, self.data.target_class]\
                     - self.data.predict_fn(previous_CF_candidate)[:, self.data.target_class]
        distance = self.distance_metric.measure(self.data.X,X_prune)\
                   -self.distance_metric.measure(self.data.X, previous_CF_candidate)
        idx_max = np.argmax(score_diff / (distance + self.data.eps)[:,np.newaxis])
        CF_candidate = X_prune[idx_max:idx_max+1, :]
        return CF_candidate

class PlausibilityReward(RewardFunction):
    def __init__(self,data, auto_encoder, **kwargs):
        self.data = data
        self.auto_encoder = auto_encoder

    def calculate_reward(self,X_prune,previous_CF_candidate):
        score_prune = self.data.predict_fn(X_prune)
        score_diff = self.data.predict_fn(X_prune)[:, self.data.target_class]\
                     - self.data.predict_fn(previous_CF_candidate)[:, self.data.target_class]#target_class for multiclass
        if previous_CF_candidate != 'None':
            if X_prune != 'None':
                AE_loss_diff = self.auto_encoder(previous_CF_candidate)-self.auto_encoder(X_prune)
            else:  print("X_prune is null")   
        else: print("previous_CF_candidate is null") 
        idx_max = np.argmax(score_diff * (AE_loss_diff)[:,np.newaxis])
        CF_candidate = X_prune[idx_max:idx_max+1, :]
        return CF_candidate


class DifferenriallPrivacyAward(RewardFunction):
    def __init__(self,data,distance_metric:DistanceMetric,**kwargs):
        self.data = data
        # self.k_neghbours = k_neighbours
        self.distance_metric = distance_metric

     #for all values from all neighbours, based on proximity selects feature value, we should add: 
        #1- change_cost(done), #
        #2-other measures except for proximity, 
        #3- consider actionability (from begining, when creating X_prune) Done 
        #4- combine proximity and change cost(done) 5- add rule set
    def calculate_reward(self,X_prune,previous_CF_candidate):      
        score_prune = self.data.predict_fn(X_prune)
        score_diff = self.data.predict_fn(X_prune)[:, self.data.target_class]\
                     - self.data.predict_fn(previous_CF_candidate)[:, self.data.target_class]
        distance = self.distance_metric.measure(self.data.X,X_prune)\
                   -self.distance_metric.measure(self.data.X, previous_CF_candidate)
        idx_max = np.argmax(score_diff / (distance + self.data.eps)[:,np.newaxis])
        CF_candidate = X_prune[idx_max:idx_max+1, :]
        return CF_candidate            
    