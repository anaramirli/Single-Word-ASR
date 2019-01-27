# Speech-Recognition

This project involves building context based speech recognition in Fligh Simulator for a limited vocabulary by utilizing the Neural Networks, [Hidden Markov Models](https://mi.eng.cam.ac.uk/~mjfg/mjfg_NOW.pdf) and the hyperparameter optimization of the outcome of the built-in models.


Dataset: 41 context based words, 30620 entries<br/><br/>
`src` directory to view notebook and python files


| Model | Train Acc | Validation Acc | Test Acc* | Rejection
| :--- | :---: | :---: | :---: | ---: |     
| HMM   | 96.12% | - | 95.12% | - |
| NN | 99.99% | 98.98% | 97.62% | 1.66% |
| NN (one vs. all) | >99.45% | >98.84% | 97.00% | 2.86% |

`* =  when h1>0.9 and h1-h2>0.5 for unseen data`


# Project Instructions

## Getting Started

1. Clone the repository, and navigate to the downloaded folder.

    ```
    git clone https://github.com/anaramirli/contex-based-asr.git
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
    requirements.py
    ```
