# Speech-Recognition

This project involves building multi-speaker speech recognition for 41 words and phrase combinations of different lengths using the Neural Networks, [Hidden Markov Models](https://mi.eng.cam.ac.uk/~mjfg/mjfg_NOW.pdf) and the hyperparameter optimization of the outcome of the built-in models.

The result of the log argmax of HMM on the *Train* and unseen *Test* setes are **96.24%** and **95.74%**, accordingly. This indicates that the method has a good generalization.

The accuracy of NN on *Valdation* and unseen *Test* setes are **98.93%** and **98.44%**, accordingly. This indicates that the method has a good generalization.

| Model | Train Acc | Validation Acc | Test Acc |
| :---         |     :---:      |     :---:      |     :---:     |     
| git status   | git status     | git status    | sth            |
| git diff     | git diff       | git diff      | adwd           |

Main implementation can be found on **HMM_speech_recognizer.ipynb** and **NN_speech_recognizer.ipynb**.
