# Speech-Recognition

This project involves building context based speech recognition in Fligh Simulator for a limited vocabulary by utilizing the Neural Networks, [Hidden Markov Models](https://mi.eng.cam.ac.uk/~mjfg/mjfg_NOW.pdf) and the hyperparameter optimization of the outcome of the built-in models.


Dataset: 41 context based words, 30620 entries<br/>
`src` directory to view notebook and python files<br/>
`documentation` directory to view documentation files.<br/>


| Model | Train Acc | Validation Acc | Test Acc* | Rejection
| :--- | :---: | :---: | :---: | ---: |     
| HMM   | 96.12% | - | 95.12% | - |
| NN | 99.99% | 98.98% | 97.62% | 1.66% |
| NN (one vs. all) | >99.45% | >98.84% | 97.00% | 2.86% |

`* =  when c1 and c2 meet`

$$c1=\overline{y}_k \geq \Delta_1$$
$c2=\overline{y}_k - \widetilde{y}_p \geq \Delta_2$

$y = \left \{ {y_0,y_1,... {y}_{N}} \right \}$ is the set of the  output for all classes

$overline{y}_k = \underset{1\leq i \leq N }{\max y_i}, \quad k = \underset{1\leq i \leq N }{\arg \max y_i}$

$\widetilde{y}_p = \underset{i\leq i\leq k ; k+1\leq i\leq N }{\max y_i}$


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
