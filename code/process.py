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

def grayscale(images):
    """
    Convert a set of images to grayscale

    Inputs
    ----------
    image: numpy.ndarray
        Aarray containing a set of RGB images

    Outputs
    -------
    grayscale_images: numpy.ndarray
        Array containing a set of grayscale images

    """      
    grayscale_images = []
    for image in images:
        grayscale_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    
    grayscale_images = np.asarray(grayscale_images)

    return grayscale_images


def normalize_images(images, a, b, image_min, image_max):
    """
    Min-max feature scaling

    Inputs
    ----------
    images: numpy.ndarray
        Array containing a set of images
    a,b: float,float
        Scaling range
    images_min, images_max: float,float
        Min/max values of 'images'

    Outputs
    -------
    normalized_images: numpy.ndarray
        Array containing a set of images scaled to [a,b]

    """ 
        
    normalized_images = a + (((images - image_min) * (b - a)) / (image_max - image_min))

    normalized_images = np.asarray(normalized_images)
    
    return normalized_image