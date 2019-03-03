# Copyright 2019 Anar Amirli

import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn import preprocessing
import itertools
import re


def atoi(text):
    return int(text) if str(text).isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', int(text) if str(text).isdigit() else text) ]

def load_model(w):
    '''
    Load model from dir
    '''
    
    model_group = None

    if (w.find('.h5')!=-1):
        # model_name=w
        # [str(model_name[:-3])]
        model_group = load_model('models/1_vs_1/mix'+'/{model}'.format(model=model_name))
            
    return model_group


def seperate_list(list, division_part):
    """
    Dividing list into equal parts or partialy equal parts
    Parameters
    ----------
    
    list - list
    division_part - number of part to divide list into
    
    Returuns
    --------
    
    out - single list containing all the divided lists
    
    """
    avg = len(list) / float(division_part)
    out = []
    last = 0.0

    while last < len(list):
        out.append(list[int(last):int(last + avg)])
        last += avg
    return out


def plot_confusion_matrix(cm,
                          classes,
                          xlabel,
                          ylabel,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    Plot the given confusion matrix cm as a matrix using and return the
    resulting axes object containing the plot.
    Parameters
    ----------
    cm : ndarray
        Confusion matrix as a 2D division_partpy.array.
    classes : list of str
        Names of classified classes.
    xlabel : str
    Label of the horizontal axis.
    ylabel : str
    Label of the vertical axis.
    normalize : bool
        If True, the confusion matrix will be normalized. Otherwise, the values
        in the given confusion matrix will be plotted as they are.
    cmap : matplotlib.colormap
        Colormap to use when plotting the confusion matrix.
    Returns
    -------
    fig : matplotlib.figure
        Plot figure.
    ax : matplotlib.Axes
        matplotlib.Axes object with horizontal bar chart plotted.
    References
    ----------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    vmin = None
    vmax = None
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        vmin = 0
        vmax = 1

    plt.figure(figsize=(20, 20))
    cax = plt.imshow(
        cm, interpolation='nearest', vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(cax)
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)


    plt.yticks(tick_marks,classes)


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        cell_str = '{:.2f}'.format(cm[i, j]) if normalize else str(cm[i, j])
        plt.text(
            j,
            i,
            cell_str,
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
  
    plt.show()
    
    

def plot_keras_los_acc_hist(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()