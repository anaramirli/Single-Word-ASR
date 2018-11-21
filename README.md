# Speech-Alignment

This documentation represents fundamental principles of speech recognition and its alignment.
We'll use Gaussian HMMs to model our data.

The original project and dataset can be found [here](https://code.google.com/archive/p/hmm-speech-recognition/downloads)

### How it works

    Written Text -> Phonological synthesizer -> Feature Extraction -> =text_features
    Natural Speech Signal-> Feature Extraction-> =speech_features
    text_features + speech_features -> DTW (Natural Speech with Phonetic Segmentation) = forced alignment

DTW (Dynamic Time Wraping) is an algorithm to align temporal sequences with possible local non-linear distortions.
