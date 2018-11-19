# Speech-Alignment

This documentation represents fundamental principles of speech recognition and its alignment.
We'll use Gaussian HMMs to model our data.

The original project and dataset can be found [here](https://code.google.com/archive/p/hmm-speech-recognition/downloads)



## Forced Speech Alignment

This demo project addres the determination of the location of phonemes in a speech signal, given the squence of phonemes constrained in that signal. 

### What is forced alignment?

Forced alignment is a technique to take an orthographic transcription of an audio file and generate a time-aligned version using a pronunciation dictionary to look up phones for words. 

### What we can do?
* Convert analogue recordings to digital format
* **Identify where is the audio speech sounds of interest are**

### Finding words in audio

Based on words phonetic frequeny it is possible to identify their place in audio speech.

### Bits and Pieces and Issues for doing forced alignment

##### Piece 1: A pronouncing dictionary

    well - W EH1
    there - DH EH1 R
    was - W AH0 Z
    one - W AH1 N
    time - T AY1 M  

##### Issue 1: Pronounciation Variants

    walking - W AO1 L K IHO N
    walking - W A01 L K IHO NG
    walking - W AO1 L K IHO NG G
            
* **Option 1**: *include all options* <br>
Let the aligner figure out which option to use
    * **Pros**
        - You'll get more accurate timing.
    * **Cons**
        - It can be tricky to identify whic pronounciations are varients of each other. 
        

* **Option 2**: *only include one option* <br>
Only allow the aligner to choose one options
    * **Pros**
        - It'll be easier to identify all instances of potential pronunciation variation.
    * **Cons**
        - The timing information will be less accurate.
    
##### Issue 2: Out of Dictionary Words

No matter how large a pronouncin dictionary you're working with, there will always be some words in free flowing speech that aren't in the dictionary.

    Fruehwald - F R UW1 W AO0 L D
    hoagie - HH OW1 G IY0
    
These either need to be added to the dictionary when the aliner is run, or a sepeate piece of software needs to try to guess the pronounciation based on the spelling.   

##### Piece 2: An acousti model

W=0.090000
W=0.030000

##### Piece 3: A transcript
Outside of the original fieldwork, this is the most time consuming and expensive part.

### How it works?

    Written Text->Phonetizer(Phonoetic Transcription) ->Feature Extraction-> =text_features
    Natural Speech->Feature Extraction-> =speech_features
    text_features + speech_features -> DTW (Natural Speech with Phonetic Segmentation) = forced alignment

DTW (Dynamic Time Wraping) is an algorithm to align temporal sequences with possible local non-linear distortions.

### Concerns about forced alignment
It will make mistakes
* It is easier and faster (read: ceaper) to manually correct the output of automated systems than to create annotations from scratch.
* Humans make mistakes too! And the kinds of mistakes automated systems make are usually *systematic*, so they're easier to identify and locate. 
