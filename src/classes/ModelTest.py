import numpy as np
import os
import pandas as pd
import json
from scipy.io import wavfile
import math
from python_speech_features import mfcc, logfbank
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# our classes
from classes.EvaluateModel import *



class ModelTest(object):
    
    eval_model_dir = dict()
    
    # normalization values
    mean = np.array
    var = np.array
    
    
    def __init__(self, model_grid):

        self.model_grid = model_grid

      
     
    def init_models(self):

        for model in self.model_grid:
            print(model['model_name'])

            # initialize evaluate model
            evaluate = EvaluateModel(model['model_name'], model['api_name'], model['model_type'], model['model_path'], model['scaler_path'],  dict_path=model['dict_path'], class_size=model['class_size'])
            # get model
            evaluate.models = evaluate.get_models()
            # append model to dict
            self.eval_model_dir[model['model_name']] = evaluate

            del evaluate
    
    def clean_zeros_features(self, X, scaler_mean_var, normalize):

        # sequence data
        X_out=[]


        for j in range(len(X)-1,0,-13):

            if(X[j]!=0):
                index = j+1
                while ((index)%13!=0):
                    index+=1

                new_data=X[0:index]

                if normalize==True:
                    mean = scaler_mean_var.values[0,:]
                    var =  scaler_mean_var.values[1,:]
                    for idx in range(index):
                        new_data[idx]=(new_data[idx]-mean[idx])/math.sqrt(var[idx])

                new_data=np.reshape(new_data,(int(index/13),13))

                X_out.insert(0,new_data)

                break

        return np.array(X_out)

    
    def get_mfcc(self, audio, scaler_mean_var, normalize, api_name, seq_lenghth, sampling_freq):
      
        
        if normalize==True and scaler_mean_var is None:
            assert False, "Sclaer values can not be null when normalize is True"
        
        
        audio = np.array(audio)
        # Extract MFCC features
        mfcc_features = mfcc(audio, sampling_freq)
        # create non-sequential mffc by converting mfcc it 2D array and then padding zeroes
        mfcc_len=mfcc_features.shape[0]*mfcc_features.shape[1]
        
        # resize flat data into 2D array
        mfcc_features = np.resize(mfcc_features,(1,mfcc_len))
        non_sequence_mfcc = np.zeros((1,seq_lenghth), dtype=float)
        
        
        # get mfcc featuress as 1D array (non-sequence)
        if (mfcc_len>seq_lenghth):
            non_sequence_mfcc[:,:] = mfcc_features[:,0:seq_lenghth]
        else :
            non_sequence_mfcc[:,0:mfcc_len]= mfcc_features


        # get mfcc features as sequence
        if "hmmlearn" in api_name.lower():
            
            sequence_mfcc = self.clean_zeros_features(X=non_sequence_mfcc[0], scaler_mean_var=scaler_mean_var, normalize=normalize)
            return sequence_mfcc    
        
        
        if normalize==True:
            mean = scaler_mean_var.values[0,:]
            var =  scaler_mean_var.values[1,:]
            for idx_nor in range(seq_lenghth):
                non_sequence_mfcc[:,idx_nor]=(non_sequence_mfcc[:,idx_nor]-mean[idx_nor])/(math.sqrt(var[idx_nor]))


        return non_sequence_mfcc
    
            
    def get_model_result(self, audio, model_name, h1=0.9, h2=0.5, normalize=False,seq_lenghth=2808, sampling_freq=16000):
        
        # get prediction results
    
        # get model
        model = self.eval_model_dir[model_name]
            
        # get mffc values 
        if normalize==True:
            data_mfcc = self.get_mfcc(audio=audio, scaler_mean_var=model.scaler_mean_var, normalize=True, api_name=model.api_name, seq_lenghth=2808, sampling_freq=16000)
        else:
            data_mfcc = self.get_mfcc(audio=audio, scaler_mean_var=None, normalize=False, api_name=model.api_name, seq_lenghth=2808, sampling_freq=16000)
            
        

        if "hmmlearn" not in (model.api_name).lower():
            
            predicted_labels = model.calculate_res(models=model.models, h1=h1, h2=h2, X=data_mfcc)

            if int(predicted_labels[0])!=41: 
                return model.model_dict[str(predicted_labels[0])]
            else:
                return 'unknown'
        else:
            predicted_labels = model.calculate_res_sequential(models=model.models, X=data_mfcc)
            #result= self.model_dict[str(predicted_labels[0])]
            return model.model_dict[str(predicted_labels[0])]
                