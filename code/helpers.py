"""
This function contains various helper functions
"""

# Imports
import numpy as np
from os import listdir
from os.path import isdir as folder_exists

def folder_is_empty(path):

    """
    Check if a folder is empty
    
    Inputs
    ----------
    path: string
        Full path to a keras model
        
    Outputs
    -------
    result: bool
        True if folder is empty, false otherwise
    """

    result = True

    if folder_exists(path):
        result = (len(listdir(path)) == 0)
    
    return result

def data_pick_subset(data, indices):
    """
    Pick a subset of a dataset
    
    Inputs
    ----------
    data: numpy.ndarray
        Array containing some kind of data
    indices: string
        Indices specifying the elements (or order of elements) of 'data' included in 'subset'
        
    Outputs
    -------
    subset: numpy.ndarray
        Array containing a subset of 'data'
    """   

    subset = []
    
    for index in indices:
        subset.append(data[index])

    subset = np.asarray(subset)

    return subset

def images_pick_subset(images, labels = None, labels_metadata = None, indices = None, n_images_max = 25):
    """
    Pick a subset of images
    
    Inputs
    ----------
    images: numpy.ndarray
        A set of images, RGB or grayscale
    labels: (None | numpy.ndarray)
        A set of image classes
    labels_metadata: (None | numpy.ndarray)
        A set of metadata about image classes
    n_images_max: int
        The maximum number of images allowed in 'images_subset'

        
    Outputs
    -------
    images_subset: numpy.ndarray
        A subset of 'images'
    labels_subset: (None | numpy.ndarray)
        A subset of 'labels'
    labels_metadata_subset: (None | numpy.ndarray)
        A subset of 'labels_metadata'
    
    """

    n_images = len(images)

    if indices is None:
        indices = np.random.randint(0, n_images, min(n_images, n_images_max))

    images_subset = data_pick_subset(images, indices)

    labels_subset = None
    if labels is not None:
        labels_subset = data_pick_subset(labels, indices)

    labels_metadata_subset = None
    if (labels is not None) and (labels_metadata is not None):
        labels_metadata_subset = data_pick_subset(labels_metadata, labels_subset)

    return images_subset, labels_subset, labels_metadata_subset

def dist_is_uniform(labels):

    """predictions_pick_subset
    Check if a label distribution is uniform
    
    Inputs
    ----------
    labels: numpy.ndarray
        A set of image classes

        
    Outputs
    -------
    is_uniform: bool
        True if 'labels' is uniform, False otherwise
    
    """
    
    classes, classes_count = np.unique(labels, return_counts = True)

    class_ref_count = classes_count[0]

    is_uniform = True

    for class_count in classes_count:

        if class_count != class_ref_count:
            is_uniform = False

    return is_uniform


def predictions_create_titles(predictions, labels, indices = None, top_k = 5, n_images_max = 25):
    """
    Create a set of image titles based on a set of predictions
    
    Inputs
    ----------
    predictions: numpy.ndarray
        A set of image class predictions from a model
    labels: numpy.ndarray
        A set of image classes
    labels_metadata: (None | numpy.ndarray)
        A set of metadata about image classes


    Outputs
    -------
    top_k_predictions: 
    
    """

    if indices is None:
        indices = np.arange(min(len(predictions), n_images_max))

    predictions = data_pick_subset(predictions, indices)

    titles = []

    for prediction in predictions:
        top_k_predictions  = prediction.argsort()[-top_k:][::-1]
        top_k_probabilities = np.sort(prediction)[-top_k:][::-1]

        labels_subset = data_pick_subset(labels, top_k_predictions)

        title = ''
        for k, prob in enumerate(top_k_probabilities):
            title += labels_subset[k] + " " + "P:" + str(prob) + "\n"

        titles.append(title)

    return titles

def web_data_file_names_are_valid(folder_path):
    """
    Verify that the web data image names are valid
    
    Inputs
    ----------
    folder_path: string
        Path to a folder containing a set of images 


    Outputs
    -------
    result: bool
        True if valid, False otherwise 
    
    """

    result = True

    file_names = sorted(listdir(folder_path))

    tmp = 0
    for file_name in file_names:
        try:
            tmp = int(file_name[:len(file_name) - 4])
        except ValueError:
            result = False
            break

    return result







    