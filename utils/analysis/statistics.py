
import pandas as pd
import numpy as np
 #To fetch a dataset from the PMLB
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
#from nice import NICE
from dpmechanisms.new_nice import NICE
# from Experiments.AutoEncoder import AeTrainer
# from Experiments.Preprocessing import PpModeler
# from Experiments.DataGathering.Fetchers import PmlbFetcher
from utils.data.preprocessing import find_feature_ranges
from dpmechanisms.local_dp import Make_Private,Compare
from dpmechanisms.linkage import reidentify,unique_instances
from utils.analysis.draw_plots import draw_CDF,draw_dist

AE_GRID = {'learning_rate_init': [0.05,0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001]}
#number of neighbors to test
#N_numbers = [1,3,10]  #correct one
N_numbers = [3,10]  ##### during debug
#different privacy cost values
#eps_list = [.01,.1,1]   ### Correct one
eps_list = [.01,.1,1,5]  #### during debug

def compare_solutions(Factual_instance,basic,CF):   #find how the CF changes in comparison with basic_cf and original instance
        eps = 0.00000000001
        # differencses between Factual_instance and basic Counterfacual
        nonoverlapping_features = np.where(Factual_instance != basic)[1]
        nonoverlapping = np.count_nonzero(nonoverlapping_features)
        #difference between Factual_instance and generated counterfactual
        Changes_features = np.where(Factual_instance != CF)[1]
        Changed = np.count_nonzero(Changes_features)
        # ratio of features selected from basic_CF   : in changed feutures, find what is equal to basic
        #Chosen_from_basic_features = np.where(Factual_instance != CF and CF == basic)[1]
        Chosen_from_basic_features = np.where(np.logical_and(Factual_instance != CF , CF == basic))[1]
        Chosen_from_basic = np.count_nonzero(Chosen_from_basic_features)
        # ratio of changes features
        changed_ratio = Changed / ( nonoverlapping + eps )  #to avoid devision by zero
        # ration of selected from basic instance
        from_basic_ratio = Chosen_from_basic / ( Changed + eps)   #to avoid devision by zero
        return changed_ratio,from_basic_ratio




adult = fetch_data('adult')

#Drop the unnecessary columns for feature set
X = adult.drop(columns=['education-num','fnlwgt','target','native-country','capital-gain','capital-loss'])
#Assign ‘target’ column with all rows as labels
y = adult.loc[:,'target']
#List of columns remaining in X will form the feature set
feature_names = list(X.columns)
costs = [.05,.05,.1,5,.1,5,10,10,.01,.01,.05]
zerocosts = [0,0,0,0,0,0,0,0,0]
print(feature_names)

X = X.values  
print(X.shape)   #Display updated X 

y=y.values
print(y.shape)   #Display updated y 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

categorical_features = [1,2,3,4,5,6,7]
numerical_features = [0,8]
changhable_features = [0,1,2,4,8]
increase_only = [0,1,2]

feature_ranges = find_feature_ranges(X_train,numerical_features,categorical_features)

# countlist =[]
# for instance in X_train:
#        #             ldp_cf_countlist[eps].append(reidentify(ldp_cf,X_train))
#         countlist.append(reidentify(instance.reshape(1,-1),X_train))

# draw_CDF(countlist)


unique_x_train , unique_y_train , count_list = unique_instances(X_train,y_train)
print("number of unique instances:",len(unique_x_train))
print("number of labels:",len(unique_y_train))
draw_CDF(count_list)
