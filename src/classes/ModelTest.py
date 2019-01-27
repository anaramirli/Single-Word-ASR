import numpy as np
import os
import pandas as pd
import json
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# our classes
from classes.EvaluateModel import *


class ModelTest(object):
    
    eval_model_dir = dict()
    
    def __init__(self, dict_path, model_grid, test_path, class_size):

        self.model_grid = model_grid
        self.class_size = class_size
        self.dict_path = dict_path
        self.test_path = test_path
        
        if type(self.class_size)!=int:
            assert False, "class_size must be integer"
        
        if not os.path.isfile(dict_path):
            assert False, "ditc file does not exists"
        else:
            # get label dict
            with open(self.dict_path, encoding='utf-8') as data_file:
                self.model_dict = json.loads(data_file.read())

        if not os.path.isfile(self.test_path):
            assert False, "test path does not exists"
        else:
            test_df = pd.read_csv(self.test_path)
            X_out = test_df.values[:,1:]
            self.scaler = StandardScaler().fit(X_out)

      

    def init_models(self):

        for model in self.model_grid:
            print(model['model_name'])

            # initialize evaluate model
            evaluate = EvaluateModel(model['model_name'], model['model_type'], model['model_dir'], class_size=self.class_size)
            # get model
            evaluate.models = evaluate.get_models()
            # append model to dict
            self.eval_model_dir[model['model_name']] = evaluate

            del evaluate
    
    
    def get_mfcc(self, wav_path, array_length=2808):
        
        
        if not os.path.isfile(wav_path):
            assert False, "File does not exists"
        else:
            
            # read file 
            sampling_freq, audio = wavfile.read(wav_path)
            # Extract MFCC features
            mfcc_features = mfcc(audio, sampling_freq)

            # normalize sequence_mfcc 
            scaler = StandardScaler().fit(mfcc_features)
            sequence_mfcc  = scaler.transform(mfcc_features)

            # create non-sequential mffc by converting mfcc it 2D array and then padding zeroes
            mfcc_len=mfcc_features.shape[0]*mfcc_features.shape[1]
            # resize flat data into 2D array
            mfcc_2d = np.resize(mfcc_features,(1,mfcc_len))
            non_sequence_mfcc = np.zeros((1,array_length), dtype=float)
            
            if (mfcc_2d.shape[1]>array_length):
                non_sequence_mfcc[0,0:]=mfcc_2d[0,0:array_length]
            else :
                non_sequence_mfcc[0,0:mfcc_2d.shape[1]]= mfcc_2d

            # normalize non-sequential
            non_sequence_mfcc=self.scaler.transform(non_sequence_mfcc)

        return sequence_mfcc, non_sequence_mfcc
    
            
    def get_model_result(self, wav_path, model_name):
        
        result = ""
    
    
        # get mffc values 
        sequence_mfcc, non_sequence_mfcc = self.get_mfcc(wav_path)
            
        # get model
        model = self.eval_model_dir[model_name]
        # get prediction results

        if model_name!="HMMs":
            predicted_labels = model.calculate_res(model.models, h1=0.9, h2=0.5, X_test=non_sequence_mfcc)

            if int(predicted_labels[0])!=41: 
                result =self.model_dict[str(predicted_labels[0])]
            else:
                result = 'unknown'
        else:
            predicted_labels = model.calculate_res_sequential(model.models, X_test=[sequence_mfcc])
            result= self.model_dict[str(predicted_labels[0])]
                
        return result