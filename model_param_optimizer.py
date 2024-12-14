import argparse
import os
import dill
from tools.save_results import save_results_to_csv
from sklearn.model_selection import train_test_split
from dpmechanisms.new_nice import NICE
from utils.data.load_data import Fetcher
from tools.models import create_model_pipeline,calculate_accuracy,optimize_model_pipeline
import json

####### these parameters are inputs of our function #######

#RANDOM_SEED = 945#,42,83,131,354,945
#NEIGHBOR_COUNT = 5
#CF_method = 'zerocost_DP_CF'# 'NICE' #'zerocostPlaus_DP_CF'
#model_name = 'SVM' #'RF', 'SVM' , 'XGBoost','NN'
#epsilon = 5
# load all data, feature names, ranges, costs and sets other variables



def model_optimizer(DS_name, model_name):

    
    # DS = Fetcher(dataset,epsilon=epsilon) #hospital,adult,informs,synth_adult
    # X= DS.dataset['X'].values
    # Y = DS.dataset['y'].values
    # #Create pipeline

    # X_train, X_test_1, y_train, y_test_1 = train_test_split(X, Y, test_size=0.3,random_state=RANDOM_SEED)
    # # here: select 1% of the dataset to check for exponential mechanism

    
    # DS.X_train = X_train.copy()
    # DS.y_train = y_train.copy()
    
    # X_test,X_counterfactual,y_test,y_counterfactual=train_test_split(X_test_1, y_test_1, test_size=0.3,random_state=RANDOM_SEED)
    # DS.X_test = X_test.copy()
    # DS.y_test = y_test.copy()
    # DS.X_counterfactual = X_counterfactual.copy()
    # DS.y_counterfactual = y_counterfactual.copy()
    

    # DSoutdir = './dpnice/datasets_loaded/{}/'.format(dataset)
    # if not os.path.exists(DSoutdir):
    #     os.makedirs(DSoutdir, exist_ok=True)
    # Note: for synthetic data train model over all epsilon values, for real datasets train model only once
    
    with open(DS_name, 'rb') as file:
        DS = dill.load(file)
                
    # if dataset == 'synth_adult' : #generate model name, train mode and save model in the generated name, this name and training contains epsilon
    #     DS_name = '{}{}_{}_{}.pkl'.format(DSoutdir, model, RANDOM_SEED,epsilon)
    # elif epsilon == 1:  #generate model name, train mode and save model in the generated name, this name and training does not contain epsilon
    #     DS_name = '{}{}_{}.pkl'.format(DSoutdir, model, RANDOM_SEED)

    # dill.dump(DS, open(DS_name,"wb"))
    print("dataset loaded:", DS_name,"\n")
    ############### Model Optimization ################
    best_model,best_params = optimize_model_pipeline(model_name, DS.dataset['numerical_features'],DS.dataset['categorical_features'], DS.X_train, DS.y_train)
    

    # prediction = lambda x: best_model.predict_proba(x)

    # accuracy = calculate_accuracy(best_model, DS.X_test, DS.y_test)
    # best_params['accuracy'] = accuracy
    return best_model,best_params
    ############## End Model Optimization ################
    #### pass best_model for training
    # clf = create_model_pipeline(model_name,DS.dataset['numerical_features'],DS.dataset['categorical_features'])
    # clf.fit(X_train, y_train)

    # # define prediction
    # prediction = lambda x: clf.predict_proba(x)

    # #Note: calculate model accuracy on test data
    # acc = calculate_accuracy(clf, X_test, y_test)

    # result_file = './dpnice/results/models_accuracy.csv'
    # model_info = {'dataset': dataset, 'RANDOM_SEED': RANDOM_SEED, 'model': model, 'epsilon': epsilon,'accuracy':acc},
    # save_results_to_csv(model_info, result_file)
    
    # # define nice model to train
    # #another input parameter is distance measure, like 
    # # what we have here is an initial version
    # NICE_model = NICE(optimization='proximity', #'differentialprivacy',  #'plausibility',    #optimization method
    #                 justified_cf=True,
    #                 X_train = X_train,   #training data
    #                 predict_fn=prediction,    #prediction function
    #                 y_train = y_train,
    #                 #if optimization = 'Plausibility'
    #                 #   auto_encoder= autoencoder,
    #                     #if optimization is 'differentialprivacy'
    #                     # set k_neighbors
    #                 #categorical features 
    #                 cat_feat=DS.dataset['categorical_features'],
    #                 #numerical features   
    #                 num_feat=DS.dataset['numerical_features']) 
    # return NICE_model


if __name__ == '__main__':

    # parser initialization
    parser = argparse.ArgumentParser(description='Script pretraining bbox models')
    parser.add_argument('--dataset', type=str, default='compas', help='hospital,adult,informs,synth_adult')
    parser.add_argument('--rseed', type=int, default=2, help='random seed: choose between 0 - 5')
    parser.add_argument('--model', type=str, default='NN', help='NN, RF, SVM, XGBoost')
    parser.add_argument('--epsilon', type=float, default='1', help='0.01, 0.1, 1, 5, 10')
    #here we do not have k and CF  and indices since we are just training our models
    
    # get input      
    args = parser.parse_args()
    dataset = args.dataset
    RANDOM_SEED = args.rseed
    model = args.model
    epsilon = args.epsilon
    
    #### initiate with test data
    # dataset = 'adult'
    # RANDOM_SEED = 2
    # model = 'RF'
    # epsilon = 10 
    # create directory to save outputs (for here, pre trained models)
    if(epsilon >= 1):
        epsilon = int(epsilon)
    resutldir =  './dpnice/model_selection/results/'
    modeloutdir = './dpnice/model_selection/{}/'.format(dataset)
    DSoutdir = './dpnice/datasets_loaded/{}/'.format(dataset)
    result_file = './dpnice/model_selection/results/model_accuracy.txt'
    if not os.path.exists(modeloutdir):
        os.makedirs(modeloutdir, exist_ok=True)
    # Note: for synthetic data train model over all epsilon values, for real datasets train model only once
    
    # if dataset == 'synth_adult' : #generate model name, train mode and save model in the generated name, this name and training contains epsilon
    #     model_name = '{}{}_{}_{}.pkl'.format(modeloutdir, model, RANDOM_SEED,epsilon)
    #     trained_model = model_optimizer(dataset,model, RANDOM_SEED,epsilon)
    #     dill.dump(trained_model, file=open(model_name,"wb"))
    # elif epsilon == 1:  #generate model name, train mode and save model in the generated name, this name and training does not contain epsilon
    best_model_name = '{}{}_{}.pkl'.format(modeloutdir, model, RANDOM_SEED)
    best_params_file = '{}{}_{}_params.json'.format(modeloutdir, model, RANDOM_SEED)
    DS_name = '{}{}_{}.pkl'.format(DSoutdir, model, RANDOM_SEED)

    best_model,best_params = model_optimizer(DS_name,model)
    
    dill.dump(best_model, file=open(best_model_name,"wb"))
        # dill.dump(trained_model, file=open(model_name,"wb"))
    print(f"Best model saved to {best_model_name}")
    
    serializable_best_params = {k: v for k, v in best_params.items() if isinstance(v, (int, float, str, bool, list, dict))}
    
    # Save best_params to a JSON file
    with open(best_params_file, 'w') as f:
        json.dump(serializable_best_params, f, indent=4)
    

    # with open(best_params_file, 'w') as f:
    #     json.dump(best_params, f, indent=4)
    
    
    print(f"Best parameters saved to {best_params_file}")
        
