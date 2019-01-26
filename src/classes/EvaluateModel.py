# Copyright 2019 Anar Amirli

import numpy as np
import os
from keras.models import load_model
from keras.models import Sequential
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from keras.layers import Dropout
from keras.layers import Dense
import pandas as pd
import pickle
import re


class EvaluateModel(object):
    
    
    # loaded models
    models = []
    
    def __init__(self, model_name, model_type, model_path, class_size):
        
        self.model_name = model_name
        self.model_type = model_type
        self.model_path = model_path
        
        self.class_size = class_size
        
        if type(self.class_size)!=int:
            assert False, "class_size must be integer"
            
     
        if not os.path.isdir(model_path):
            assert False, "Path does not exists"
       
        if(model_type!="normal" and model_type!="onevsall"):
            raise TypeError("Model type is not correct")
            
            
    def atoi(self,text):
            return int(text) if str(text).isdigit() else text

    def natural_keys(self,text):
        return [ self.atoi(c) for c in re.split('(\d+)', int(text) if str(text).isdigit() else text) ]

    def get_models(self):

        '''
        folder directory return get models 

        Parameters
        ----------
        dirlist: path of model directory

        '''

        models=[]
        
        files_list = os.listdir(self.model_path)
        files_list.sort(key=self.natural_keys)

        for w in files_list:
        # check wheter files is .h5 or no
            if (w.find('.h5')!=-1):
                model_name=w
                models.append(load_model(self.model_path+'/{model}'.format(model=model_name)))
            
            elif (w.find('.pkl')!=-1):
                model_name=w
                models.append(pickle.load(open(self.model_path+'/{model}'.format(model=model_name), 'rb')))

        if len(models)==0:
            assert False, "No model found"

        return models

    
    def calculate_res_sequential(self, models, X_test, argmax_axis=0):
        
        
        if type(models)!=list and len(models)<=1:
            assert False, "models must be list"
            
        if type(X_test)!=list and len(X_test)<=1:
            assert False, "X_test must be list"
        
        logprob = np.array([[m.score(i, [i.shape[0]]) for i in X_test] for m in models])
        predicted_label = np.argmax(logprob, axis=argmax_axis)
        
        return predicted_label
    
    def calculate_res(self, models, h1, h2, X_test, target=np.array([])):

        '''
        how to evaluate evaluate model:
            select the ones with larget prob.
            if prob of ones are euqal for any model then select the model with min others prob

        Pareamters
        ----------
        h1: first threshold value (used for prob>h1)
        h2: second threschold value (used for prob>h2-h1)
        X_test: x data
        target: 1D target data, if target is empty, then model works as evaluation model, not as test
        newdata: do not produce accuracy rate of finding, default False

        return
        ------
        result: nx1D array saves prediction result for each data [true/false]
        predicted_label: nx1D array save predicted labels for each data 
        '''
        
        # ignore zero division
        np.seterr(divide='ignore')
        
        actual_label = target

        # generate empty reult array
        result=np.zeros((X_test.shape[0]), dtype=int)

        # array for storing predicted labels
        predicted_label = np.zeros((X_test.shape[0]), dtype=int)

        if(self.model_type!="normal" and self.model_type!="onevsall"):
            assert False, "Model type is not correct"

        if type(models)!=list:
            assert False, "model type should be list"
        
        if(self.model_type=="onevsall" and len(models)<=1):
            assert False, "If model is onevsall, then model must be list"
            
        if(self.model_type=="normal" and len(models)!=1):
            assert False, "If model is normal, then model size must be one"

        if (self.model_type=="normal"):
            prob = models[0].predict_proba(X_test)
        else:
            prob = np.array([m.predict_proba(X_test)[:,0] for m in models]) ## prob of one gorup
            prob2 = np.array([m.predict_proba(X_test)[:,1] for m in models]) ## prob of others group
            
            prob=prob.T
            prob2=prob2.T

        for i in range(X_test.shape[0]):
            max_array=[]
            max_n=-100
            max_min=100
            idx=0
            for j in range(self.class_size):
                max_array.append(prob[i][j])

                
                if prob[i][j]>max_n:

                    max_n=prob[i][j]
                    idx=j

                    # for one-vs-all dat also calculate prob of other class
                    if (self.model_type=="onevsall"):
                        max_min=prob2[i][j]
                        
                # check h2 threshold condition
                if (self.model_type=="onevsall"):
                    if prob[i][j]==max_n and prob2[i][j]<max_min:
                        max_n=prob[i][j]
                        max_min=prob2[i][j]
                        idx=j

            # sort max array
            max_array.sort()


            if len(target)>0:
                # compare result with the actual labels
                if(int(actual_label[i])==int(idx) and max_array[-1]>=h1 and max_array[-1]-max_array[-2]>=h2):
                    result[i]=1
               
            if max_array[-1]>=h1 and max_array[-1]-max_array[-2]>=h2:
                predicted_label[i]=idx
            else:
                predicted_label[i]=self.class_size

        if len(target)>0:
            return result, predicted_label
        else:
            return predicted_label
    
    
    
    def calculate_conf_mat(self, target, predicted_labels):
        
        '''
        Calculate confusion matric including rejection

        Parameters
        ---------
        target: 1D target test data
        predicted_labels: nx1D array save predicted labels for each data 
        
        Returun
        -------
        conf_mat:
        '''
        
        conf_mat = np.zeros((self.class_size+1, self.class_size+1))
        conf_mat= conf_mat.astype(int)


        for i in range(len(target)):
            conf_mat[int(target[i])][int(predicted_labels[i])] +=1
            
        return conf_mat
    
    
    def get_rejection_ratio(self, conf_mat):
        '''
        Return rejection ratio for each classes
        
        Parameters
        ----------
        conf_mat: confussion matric
        
        Return:
        
        total_rejection: (int) total rejection ratio
        rejection: (array) rejection ratio for each classes 
        
        '''
        
        rejection=[]
        for i in range(self.class_size):
            rejection.append(conf_mat[i][self.class_size]/conf_mat[i].sum())
            
        
        total_rejection = conf_mat[:self.class_size,self.class_size:self.class_size+1].sum() / conf_mat[:self.class_size, :self.class_size].sum()
            
        return rejection, total_rejection