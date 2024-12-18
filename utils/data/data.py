from dataclasses import dataclass
from abc import ABC
import numpy as np
import sys
sys.path.append('F:\PhD\pythonprojects\dpnice')

from utils.data.distance import *
from scipy.stats import mode


class data_NICE:
    def __init__(self,X_train,y_train,cat_feat,num_feat,predict_fn,justified_cf,eps):
        self.X_train = X_train
        self.y_train = y_train
        self.cat_feat = cat_feat
        self.num_feat = num_feat
        self.predict_fn = predict_fn
        self.justified_cf = justified_cf
        self.eps = eps

        if self.num_feat == 'auto':
            self.num_feat = [feat for feat in range(self.X_train.shape[1]) if feat not in self.cat_feat]

        self.X_train = self.num_as_float(self.X_train)

        self.train_proba = predict_fn(X_train)
        self.n_classes = self.train_proba.shape[1]
        self.X_train_class = np.argmax(self.train_proba, axis=1)

        if self.justified_cf:
            self.candidates_mask = self.y_train == self.X_train_class
        else:
            self.candidates_mask = np.ones(self.X_train.shape[0],dtype=bool)




    def num_as_float(self,X:np.ndarray)->np.ndarray:
        X[:, self.num_feat] = X[:, self.num_feat].astype(np.float64)
        return X

    def fit_DS_to_X(self,X,target_class,DS):   
        self.X = self.num_as_float(X)
        self.X_score = self.predict_fn(self.X) 
        self.X_class =  self.X_score.argmax()   # predict class of X
        if target_class == 'other':
            self.target_class = [i for i in range(self.n_classes) if (i != self.X_class)]    #return a list of all classes other than class of the X
        else:
            self.target_class = target_class        # the target class is specified
        
        
        train_proba = self.predict_fn(DS.X_train)
        n_classes = train_proba.shape[1]
        DS.X_train_class = np.argmax(train_proba, axis=1)

        if self.justified_cf:
            candidates_mask = DS.y_train == DS.X_train_class
        else:
            candidates_mask = np.ones(DS.X_train.shape[0],dtype=bool)
        
        

        class_mask = np.array([i in self.target_class for i in DS.X_train_class])#todo check if this is correct for muliticlass    # generate a list of classess (other than x_class) that have instances in training set
        mask = class_mask&candidates_mask
        candidates_view = DS.X_train[mask,:].view() 
        return candidates_view

    def fit_to_X(self,X,target_class):   
        self.X = self.num_as_float(X)
        self.X_score = self.predict_fn(self.X) 
        self.X_class =  self.X_score.argmax()   # predict class of X
        if target_class == 'other':
            self.target_class = [i for i in range(self.n_classes) if (i != self.X_class)]    #return a list of all classes other than class of the X
        else:
            self.target_class = target_class        # the target class is specified
        self.class_mask = np.array([i in self.target_class for i in self.X_train_class])#todo check if this is correct for muliticlass    # generate a list of classess (other than x_class) that have instances in training set
        self.mask = self.class_mask&self.candidates_mask
        self.candidates_view = self.X_train[self.mask,:].view()       #a list of indices of training instances that belong to target class in train set

class data_SEDC:
    def __init__(self,X_train,predict_fn,cat_feat,num_feat):
        self.X_train =X_train
        self.predict_fn = predict_fn
        self.cat_feat = cat_feat
        self.num_feat = num_feat

        if self.num_feat == 'auto':
            self.num_feat = [feat for feat in range(self.X_train.shape[1]) if feat not in self.cat_feat]
        self.X_train = self.num_as_float(self.X_train)

    def num_as_float(self, X):
        X[:, self.num_feat] = X[:, self.num_feat].astype(np.float64)
        return X

    def fit(self):
        self.replace_values = np.zeros(self.X_train.shape[1])
        self.replace_values[self.cat_feat] = mode(self.X_train[:, self.cat_feat],axis=0,nan_policy='omit')[0]
        self.replace_values[self.num_feat] = self.X_train[:, self.num_feat].mean(axis = 0)
        self.replace_values = self.replace_values[np.newaxis,:]

    def fit_to_X(self,X,target_class):
        self.X=X
        self.X_score = self.predict_fn(self.X)
        self.X_class = self.X_score.argmax()
        self.n_classes = self.X_score.shape[1]
        if target_class == 'other':
            self.target_class = [i for i in range(self.n_classes) if (i != self.X_class)]
        else:
            self.target_class = target_class




