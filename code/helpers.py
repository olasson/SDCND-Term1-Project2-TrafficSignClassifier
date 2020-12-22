import numpy as np

def data_pick_subset(data, indices):
    """
    Pick a subset of a dataset
    
    Inputs
    ----------
    data : numpy.ndarray
        Array containing some kind of data
    indices: string
        Indices specifying the elements (or order of elements) of 'data' included in 'subset'
        
    Outputs
    -------
    data_restructured : numpy.ndarray
        Array containing a restructured version of 'data'
    """   

    subset = []
    
    for index in indices:
        subset.append(data[index])

    subset = np.asarray(subset)

    return subset

def images_pick_subset(images, labels = None, labels_metadata = None, indices = None, n_samples = 25):
    """
    Pick a subset of images
    
    Inputs
    ----------
    images : numpy.ndarray
        A set of images, RGB or grayscale
    labels: (None | numpy.ndarray)
        A set of image classes
    labels_metadata: (None | numpy.ndarray)
        A set of metadata about image classes
    indices: (None | numpy.ndarray)
        
    Outputs
    -------
    plt.figure
        Figure showing 'images' in an (n_rows x n_cols) layout
    
    """

    if indices is None:
        indices = np.random.randint(0, len(images), min(len(images), n_samples))

    images_subset = data_pick_subset(images, indices)

    labels_subset = None
    if labels is not None:
        labels_subset = data_pick_subset(labels, indices)

    if (labels is not None) and (labels_metadata is not None):
        labels_metadata_subset = data_pick_subset(labels_metadata, labels)

    return images_subset, labels_subset, labels_metadata_subset



    