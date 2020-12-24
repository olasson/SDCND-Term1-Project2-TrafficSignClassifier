import numpy as np
import cv2

def histogram_equalization(images):
    """
    Apply histogram equalization to a set of images

    Inputs
    ----------
    imagse: numpy.ndarray
        Array containing a set of RGB images

    Outputs
    -------
    equalized_image: numpy.ndarray
        Array containing a set of RGB images with CLAHE applied

    """   

    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (4,4))

    
    equalized_images = []
    for image in images:
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0])
        equalized_images.append(cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB))

    equalized_images = np.asarray(equalized_images)
    
    return equalized_images

def grayscale():
    pass

def normalization():
    pass