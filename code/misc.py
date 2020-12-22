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