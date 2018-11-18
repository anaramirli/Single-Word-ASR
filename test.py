import os
import warnings

import numpy as np
from scipy.io import wavfile 
from hmmlearn import hmm
from python_speech_features import mfcc


# class to handle HMM processing
class HMMTrainer(object):  
    '''
    Parameters
    ----------
    
    n_components: parameter defines the number of hidden states
    cov_type: defines the type of covariance in transition matrix
    n_iter: indicates the number of iterations for traning
    
    Choice of parameters depends on the data. 
    '''
    def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000):
        
        # initialize
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = []


        # define model
        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components, 
                    covariance_type=self.cov_type, verbose=True, n_iter=self.n_iter)
        else:
            raise TypeError('Invalid model type')
            
            
    # train data is 2D aray, where each frow is k-dimensions
    def train(self, X):
        
        # ingonre divisin by 0
        np.seterr(all='ignore')
        
        self.models.append(self.model.fit(X))
        
    # run the model on input data and get score
    def get_score(self, input_data):
        return self.model.score(input_data)
    
    
if __name__ == '__main__':

    # init the variables of all HMM models
    hmm_models = []
    
    # define train folder
    dataset = 'train'
    input_folder = 'data\{}'.format(dataset)
    
    
    # get path of input files
    try:
        audiofiles = os.listdir(input_folder)
    except FileNotFoundError:
        assert False, "Folder not found"

    # pars the input directory that contains audio files
    for dirname in audiofiles:
        # get the name of the subfolder
        subfolder = os.path.join(input_folder, dirname)
        
        if not os.path.isdir(subfolder):
            continue
        
        print(subfolder)
        
        
        # extract the label
        label = subfolder[subfolder.rfind('\\')+1:]
        
        # initialize intput variables and labels
        X = np.array([])
        y_words = []
        
        # iterate through the audio files
        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')]:
            # Read the input file
            filepath = os.path.join(subfolder, filename)
            sampling_freq, audio = wavfile.read(filepath)
            
            # Extract MFCC features
            mfcc_features = mfcc(audio, sampling_freq)

            # Append to the variable X
            if len(X) == 0:
                X = mfcc_features
            else:
                X = np.append(X, mfcc_features, axis=0)
            
            # Append the label
            y_words.append(label)
            
        # Train and save HMM model
        print ('X.shape ='+ str(X.shape))
        hmm_trainer = HMMTrainer()
        hmm_trainer.train(X)
        hmm_models.append((hmm_trainer, label))
        hmm_trainer = None
        
        
    # define test folder
    dataset = 'test'
    input_folder = 'data\{}'.format(dataset)
    
    # test folder
    test_files = []

    # get path of test files
    try:
        audiofiles = os.listdir(input_folder)
    except FileNotFoundError:
        assert False, "Folder not found"

    # pars the input directory that contains audio files
    for dirname in audiofiles:

        # get the name of the subfolder
        subfolder = os.path.join(input_folder, dirname)

        if not os.path.isdir(subfolder):
                continue

        # iterate through the audio files
        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')]:
            # read the input file

            filepath = os.path.join(subfolder, filename)
            test_files.append(filepath)      
            
            
    # Classify input data
    for input_file in test_files:

        sampling_freq, audio = wavfile.read(input_file)

        # Extract MFCC features
        mfcc_features = mfcc(audio, sampling_freq)

        # Define variables
        max_score = 0
        output_label = None

        # Iterate through all HMM models and pick 
        # the one with the highest score
        for item in hmm_models:
            hmm_model, label = item
            score = abs(hmm_model.get_score(mfcc_features))
            if score > max_score:
                max_score = score
                output_label = label
                
        print ("\nTrue:"+ input_file[input_file.find('\\')+6:input_file.rfind('\\')])
        print ("Predicted:"+ str(output_label))