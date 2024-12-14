#written by meghana
import torch.nn as nn
import torch.nn.functional as F
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import json

class SimpleNet(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def load_best_params(file_path):
    with open(file_path, 'r') as f:
        best_params = json.load(f)
    return best_params

def calculate_accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def create_model_pipeline(model_name,best_params,numerical_features,categorical_features):
    # Define your dataset and features
    # Assuming DS.dataset['numerical_features'] and DS.dataset['categorical_features'] are defined
    
    # Define the preprocessing steps in the ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # Define the model based on the provided model_name
    if model_name == 'RF': #randomforest
        model = RandomForestClassifier(**best_params)
    elif model_name == 'NN': 
        model =  MLPClassifier(**best_params)#SimpleNet(input_dim)
    elif model_name == 'SVM':
        model = SVC(**best_params)
    elif model_name == 'XGBoost':
        model = XGBClassifier()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Create the pipeline
    pipeline = Pipeline([
        ('PP', preprocessor),
        (model_name, model)
    ])

    return pipeline


def optimize_model_pipeline(model_name, numerical_features, categorical_features, X_train, y_train):
    # Create the model pipeline
    pipeline = create_model_pipeline(model_name, numerical_features, categorical_features)

    # Define the parameter grid for each model
    param_grid = {}
    if model_name == 'RF':
        param_grid = {
            'RF__n_estimators': [100], # [100, 200, 300],
            'RF__max_depth': [10] ,# [None, 10, 20, 30],
            'RF__min_samples_split': [2] # [2, 5, 10]
        }
    elif model_name == 'NN':
        param_grid = {
            'NN__hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'NN__activation': ['relu', 'tanh'],
            'NN__solver': ['adam', 'sgd'],
            'NN__alpha': [0.0001, 0.001, 0.01],
            'NN__learning_rate': ['constant', 'adaptive']
        }
    elif model_name == 'SVM':
        param_grid = {
            'SVM__C': [0.1, 1, 10, 100],
            'SVM__gamma': [1, 0.1, 0.01, 0.001],
            'SVM__kernel': ['linear', 'rbf']
        }
    elif model_name == 'XGBoost':
        param_grid = {
            'XGBoost__n_estimators': [100, 200, 300],
            'XGBoost__learning_rate': [0.01, 0.1, 0.2],
            'XGBoost__max_depth': [3, 5, 7],
            'XGBoost__subsample': [0.8, 0.9, 1.0]
        }
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Use GridSearchCV to find the best parameters
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and the best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print(f"Best parameters for {model_name}: {best_params}")
    return best_model,best_params
