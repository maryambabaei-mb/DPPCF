import argparse
import os
import dill
import json
import sys
from tools.save_results import save_results_to_csv
from sklearn.model_selection import train_test_split
from dpmechanisms.new_nice import NICE
from utils.data.load_data import Fetcher
from tools.new_models import create_model_pipeline,calculate_accuracy
from mpi4py import MPI



####### these parameters are inputs of our function #######

#RANDOM_SEED = 945#,42,83,131,354,945
#NEIGHBOR_COUNT = 5
#CF_method = 'zerocost_DP_CF'# 'NICE' #'zerocostPlaus_DP_CF'
#model_name = 'SVM' #'RF', 'SVM' , 'XGBoost','NN' 
#epsilon = 5
# load all data, feature names, ranges, costs and sets other variables

synth_Dss = ['synth_adult','synth_compas','synth_default_credit']
real_synth_map = {
    'synth_adult': 'adult',
    'synth_compas': 'compas',
    'synth_default_credit': 'default_credit'
}

def split(container, count):
    return [container[_i::count] for _i in range(count)]


def load_best_params(file_path,model_name):
    with open(file_path, 'r') as f:
        best_params = json.load(f)
    if model_name == 'SVM':
        best_params['probability'] = True
    return best_params

def train_model(dataset,model_name, RANDOM_SEED,epsilon):

    if dataset in real_synth_map:
        real_dataset = real_synth_map[dataset]
        best_params_file = f'./dpnice/model_selection/{real_dataset}/{model_name}_params.json'
    else:
        best_params_file = f'./dpnice/model_selection/{dataset}/{model_name}_params.json'

    # best_params_file = f'./dpnice/model_selection/{dataset}/{model_name}_params.json'

    DS = Fetcher(dataset,epsilon=epsilon) #hospital,adult,informs,synth_adult
    X= DS.dataset['X'].values
    Y = DS.dataset['y'].values
    #Create pipeline

    X_train, X_test_1, y_train, y_test_1 = train_test_split(X, Y, test_size=0.3,random_state=RANDOM_SEED)
    # here: select 1% of the dataset to check for exponential mechanism

    
    DS.X_train = X_train.copy()
    DS.y_train = y_train.copy()
    
    X_test,X_counterfactual,y_test,y_counterfactual=train_test_split(X_test_1, y_test_1, test_size=0.3,random_state=RANDOM_SEED)
    DS.X_test = X_test.copy()
    DS.y_test = y_test.copy()
    DS.X_counterfactual = X_counterfactual.copy()
    DS.y_counterfactual = y_counterfactual.copy()

    DSoutdir = './dpnice/optimized/datasets_loaded/{}/'.format(dataset)
    if not os.path.exists(DSoutdir):
        os.makedirs(DSoutdir, exist_ok=True)
    # Note: for synthetic data train model over all epsilon values, for real datasets train model only once
    
    if dataset in synth_Dss: #== 'synth_adult' : #generate model name, train mode and save model in the generated name, this name and training contains epsilon
        DS_name = '{}{}_{}_{}.pkl'.format(DSoutdir, model, RANDOM_SEED,epsilon)
    elif epsilon == 1:  #generate model name, train mode and save model in the generated name, this name and training does not contain epsilon
        DS_name = '{}{}_{}.pkl'.format(DSoutdir, model, RANDOM_SEED)

    dill.dump(DS, open(DS_name,"wb"))
    print("dataset loaded:", DS_name,"\n")
    best_params = load_best_params(best_params_file,model)
    clf = create_model_pipeline(model_name,best_params,DS.dataset['numerical_features'],DS.dataset['categorical_features'])
    clf.fit(X_train, y_train)

    # define prediction
    prediction = lambda x: clf.predict_proba(x)

    #Note: calculate model accuracy on test data
    acc = calculate_accuracy(clf, X_test, y_test)

    result_file = './dpnice/optimized/results/models_accuracy.csv'
    model_info = {'dataset': dataset, 'RANDOM_SEED': RANDOM_SEED, 'model': model, 'epsilon': epsilon,'accuracy':acc},
    save_results_to_csv(model_info, result_file)
    
    # define nice model to train
    #another input parameter is distance measure, like 
    # what we have here is an initial version
    NICE_model = NICE(optimization='proximity', #'differentialprivacy',  #'plausibility',    #optimization method
                    justified_cf=True,
                    X_train = X_train,   #training data
                    predict_fn=prediction,    #prediction function
                    y_train = y_train,
                    #if optimization = 'Plausibility'
                    #   auto_encoder= autoencoder,
                        #if optimization is 'differentialprivacy'
                        # set k_neighbors
                    #categorical features 
                    cat_feat=DS.dataset['categorical_features'],
                    #numerical features   
                    num_feat=DS.dataset['numerical_features']) 
    
    return NICE_model


if __name__ == '__main__':

    # parser initialization
    parser = argparse.ArgumentParser(description='Script pretraining bbox models')
    parser.add_argument('--dataset', type=str, default='synth_default_credit', help='hospital,adult,informs,synth_adult,synth_hospital,synth_informs,compas,default_credit,synth_compas')
    parser.add_argument('--rseed', type=int, default=3, help='random seed: choose between 0 - 5')
    parser.add_argument('--model', type=str, default='SVM', help='NN, RF, SVM, XGBoost')
    parser.add_argument('--epsilon', type=float, default='10', help='0.01, 0.1, 1, 5, 10')
    # parser.add_argument('--seeds', type=str, default="0,1,2,3,4", help='random seed: choose between 0 - 5')

    #here we do not have k and CF  and indices since we are just training our models
    
    # get input      
    args = parser.parse_args()
    dataset = args.dataset
    RANDOM_SEED = args.rseed
    model = args.model
    epsilon = args.epsilon
    
    # seeds = args.seeds.split(',')
    # s_seeds = [int(val) for val in seeds]
    
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # size = comm.Get_size()

    # if comm.rank == 0:

    #     jobs = split(s_seeds, comm.size)
    #     print("process {} splitted jobs:",comm.rank," Jobs: ", jobs)
    # else:
    #     jobs = None 

    #     # the root process scatters jobs
    #     jobs = comm.scatter(jobs, root=0)

    # Convert the input values to integers (or float as needed)
    
    

    #### initiate with test data
    # dataset = 'adult'
    # RANDOM_SEED = 2
    # model = 'RF'
    # epsilon = 10 
    # create directory to save outputs (for here, pre trained models)
    if(epsilon >= 1):
        epsilon = int(epsilon)
    resutldir =  './dpnice/optimized/results/'
    modeloutdir = './dpnice/optimized/pretrained/{}/'.format(dataset)
    DSoutdir = './dpnice/optimized/datasets_loaded/{}/'.format(dataset)
    result_file = './dpnice/optimized/results/model_accuracy.txt'
    if not os.path.exists(modeloutdir):
        os.makedirs(modeloutdir, exist_ok=True)
    # Note: for synthetic data train model over all epsilon values, for real datasets train model only once
    
    # for job in jobs:
    #     if dataset in synth_Dss: # == 'synth_adult' : #generate model name, train mode and save model in the generated name, this name and training contains epsilon
    #         model_name = '{}{}_{}_{}.pkl'.format(modeloutdir, model, job,epsilon)
    #         trained_model = train_model(dataset,model, job,epsilon)
    #         dill.dump(trained_model, file=open(model_name,"wb"))
    #     elif epsilon == 1:  #generate model name, train mode and save model in the generated name, this name and training does not contain epsilon
    #         model_name = '{}{}_{}.pkl'.format(modeloutdir, model, job)
    #         trained_model = train_model(dataset,model, job,epsilon)
    #         dill.dump(trained_model, file=open(model_name,"wb"))


    
    if dataset in synth_Dss: # == 'synth_adult' : #generate model name, train mode and save model in the generated name, this name and training contains epsilon
        model_name = '{}{}_{}_{}.pkl'.format(modeloutdir, model, RANDOM_SEED,epsilon)
        trained_model = train_model(dataset,model, RANDOM_SEED,epsilon)
        dill.dump(trained_model, file=open(model_name,"wb"))
    elif epsilon == 1:  #generate model name, train mode and save model in the generated name, this name and training does not contain epsilon
        model_name = '{}{}_{}.pkl'.format(modeloutdir, model, RANDOM_SEED)
        trained_model = train_model(dataset,model, RANDOM_SEED,epsilon)
        dill.dump(trained_model, file=open(model_name,"wb"))

    
        
