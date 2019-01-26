# Copyright 2019 Anar Amirli

import numpy as np
import os
import pickle
import pandas as pd
from sklearn.utils import shuffle
from keras.models import load_model
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


class PreprocessData:
    
    def preprocess_data(self, data_df, normalize=True):
        
        """
        get data and return preprocessed data

        Parameters
        ----------
        train_df: train data
        test_df: test data
        normalize: normalize data, default True;

        Return
        ------
        X_out, y_out

        """

        # get train label and data
        y_out = data_df.values[:,0]
        
        X_out = data_df.values[:,1:]

        if(normalize):
            # noramlize train 
            scaler = preprocessing.StandardScaler().fit(X_out)
            X_out=scaler.transform(X_out)

        # shuffle data
        X_out, y_out = shuffle(X_out, y_out, random_state=42)

        return X_out, y_out
    
    
    def categorize_y(self, y, target_size):
        """
        # prepare categorical target values (y) (e.g [0,0,0,1,0])

        Paramaters
        ----------
        y_data: target data

        """
        
        target = np.zeros((len(y),target_size),dtype=int)
        for i,_ in enumerate(y):
            target[i][int(_)]=1
        
        return target
    
    def uncategorize_y(self, y, axis=1):
        # untageroize data with arg max
        
        return np.argmax(y, axis)