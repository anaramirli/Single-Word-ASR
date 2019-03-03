# Speech-Recognition

This project involves building context based speech recognition in Fligh Simulator for a limited vocabulary by utilizing the Neural Networks, [Hidden Markov Models](https://mi.eng.cam.ac.uk/~mjfg/mjfg_NOW.pdf) and the hyperparameter optimization of the outcome of the built-in models.


Dataset: 41 context based words, 30620 entries<br/>
Dataset-2: 84 context based words, 38602 entries<br/>

`src` directory to view notebook and python files<br/>
`documentation` directory to view documentation files.<br/>

Current models:

| Model | Train Acc | Validation Acc | Test Acc* | Rejection
| :--- | :---: | :---: | :---: | ---: |     
| HMM-41   | 96.12% | - | 95.12% | - |
| NN-41 | 99.99% | 98.98% | 97.62% | 1.66% |
| NN-41 (one vs. all) | >99.45% | >98.84% | 97.00% | 2.86% |
| NN-41 (one vs. one) | - | - | - | - |
| HMM-84 (one vs. one) | - | - | - | - |
| NN-84 | - | - | - | - |

`* =  when h1>=0.9 and h1-h2>=0.5`

You can found more detailed information at documentation folder.

# Project Instructions

## Getting Started

1. Clone the repository, and navigate to the downloaded folder.

    ```
    git clone https://github.com/anaramirli/flight-simulator-asr.git
    cd contex-based-asr
    ```
    
2. Create (and activate) a new environment with Python 3.6.

    * **Linux** or **Mac**:
    ```
    conda create --name my_env python=3.6
    source activate my_env
    ```
    
    * **Windows**:
    
    ```
    conda create --name my_env python=3.6
    activate my_env
    ```

3. Check requiremenets.
    ```
    python requirements.py
    ```
4. Testing

    You can find implementation samples on test.py
    ```
    python test.py
    ```
5. How to use.
    
    ```python
    # call our classes
    from classes.EvaluateModel import *
    from classes.ModelTest import *
    
    
    # create model gird
    model_grid = [
        {      
            'model_name': "1_all_NN-normalize", # define model name
            'api_name': 'sequential', # api name (sequential, hmmlearn, ..)
            'model_type': "onevsall", # model type (normal, onevsall, onevsone)
            'model_path': "models/1_all_NNs/mix", # model directory
            'scaler_path': "scaler_values.csv", # scaler values - stores mean and var
            'dict_path': 'dict41.txt', # label dictionary
            'class_size': 41 # class size
        },
        
        {      
            'model_name': "HMMs", 
            'api_name': 'hmmlearn',
            'model_type': "onevsall", 
            'model_path': "models/HMMs/mix",
            'scaler_path': "scaler_values.csv",
            'dict_path': 'dict41.txt',
            'class_size': 41
        },
        
        .
        .
        .
    ]
    
    # initialize model
    model_test = ModelTest(model_grid=model_grid)
    model_test.init_models()
    
    
    # call result
    # audio is 1D array audio array, dtype=float32

    label_nn = model_test.get_model_result(audio=audio, model_name="1_all_NN-normalize", h1=0.9, h2=0.5, normalize=True, seq_length=2808, sampling=16000)
    labe_hmm =  model_test.get_model_result(audio=audio, "HMMs", normalize=True)
    
    ```
