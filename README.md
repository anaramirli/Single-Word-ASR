# Speech-Recognition

This project involves building multi-speaker speech recognition for 41 words and phrase combinations of different lengths using the Neural Networks, [Hidden Markov Models](https://mi.eng.cam.ac.uk/~mjfg/mjfg_NOW.pdf) and the hyperparameter optimization of the outcome of the built-in models.


Dataset: 41 context based words, 21432 entries<br/>
`src` directory to view notebooks


| Model | Train Acc | Validation Acc | Test Acc |
| :--- | :---: | :---: | ---: |     
| HMM   | 96.24% | - | 95.74% |
| NN | 99.99% | 98.98% | 98.79% |
| NN (one vs. all) | >99.45% | >99.84% | 98.60% |

**Accuracy on unseen test data (normal model)**
* Normal Accuracy: 98.79%
* Accuracy when (h1>=0.9): 98.38%
* Accuracy when (h1>=0.9 and h1-h2>=0.5): 98.38%

**Accuracy on unseen test data (one vs. all)**
* Normal Accuracy: 98.60%
* Accuracy when (h1>=0.9): 98.29%
* Accuracy when (h1>=0.9 and h1-h2>=0.5): 97.39%
