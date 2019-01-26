# Speech-Recognition

This project involves building context based speech recognition for 41 words and phrase combinations of different lengths using the Neural Networks, [Hidden Markov Models](https://mi.eng.cam.ac.uk/~mjfg/mjfg_NOW.pdf) and the hyperparameter optimization of the outcome of the built-in models.


Dataset: 41 context based words, 21432 entries<br/>
`src` directory to view notebooks


| Model | Train Acc | Validation Acc | Test Acc* | Rejection
| :--- | :---: | :---: | :---: | ---: |     
| HMM   | 96.24% | - | 95.74% | - |
| NN | 99.99% | 98.98% | 97.69% | 1.66% |
| NN (one vs. all) | >99.45% | >99.84% | 97.00% | 2.86% |

`* =  wehn h1>0.9 and h1-h12>0.5 for unseen data`
