"""
This file contains some basic load and save functions
"""

import matplotlib.image as mpimg
import numpy as np
import pickle

from os import listdir
from os.path import join

def data_load_pickled(path):
    """
    Load contents of a pickled file into memory
    
    Inputs
    ----------
    path : string
        Path to the pickled file
        
    Outputs
    -------
    images: numpy.ndarray
        Array containing the set of images
    labels: numpy.ndarray
        Array containing images labels
    """

    with open(path, mode = 'rb') as f:
        data = pickle.load(f)

    images = data['features']
    labels = data['labels']

    return images, labels


def data_save_pickled(path, images, labels):
    """
    Save data as a pickled file
    
    Inputs
    ----------
    path : string
        Path to where pickled file will be stored
    images : numpy.ndarray
        A set of images
    labels: numpy.ndarray
        A set of image labels
        
    Outputs
    -------
        A pickled file located at 'path'
    """

    data = {'features': images, 'labels': labels} 

    with open(path, mode = 'wb') as f:   
        pickle.dump(data, f, protocol = pickle.HIGHEST_PROTOCOL)

def data_load_web(folder_path):

    """
    Load a set of images from a folder into memory

    Inputs
    ----------
    folder_path : string
        Path to a folder containing a set of images
    Outputs
    -------
    images: numpy.ndarray
        Array containing 'images'
    """

    file_names = sorted(listdir(folder_path))
    images = []
    labels = []
    for file_name in file_names:
        images.append(mpimg.imread(join(folder_path, file_name)))
        labels.append(int(file_name[:len(file_name) - 4]))

    images = np.array(images)
    labels = np.array(labels)

    return images, labels
