"""
This file contains the data augmentation used in the project
"""

# Imports
import numpy as np
import cv2

# ---------- Constants ---------- #

# Padding used to ensure image size preservation
BORDER_PAD = 10

# Min/max for scale_image()
MIN_SCALE = 0.75
MAX_SCALE = 1.25

# Min/max for translate_image()
MIN_T = -4
MAX_T = 4

# Min/max for perspective_transform()
MIN_BORDER_OFFSET = -3
MAX_BORDER_OFFSET = 3

# Min/max for rotate_image()
MIN_ANGLE = -15
MAX_ANGLE = 15


# ---------- Functions  ---------- #

# Mirroring

def augment_data_by_mirroring(images, labels, mirror_map, n_new_images_max = 1000):
    """
    Augment a data set by creating new images by mirroring existing images

    Inputs
    ----------
    images : numpy.ndarray
        A set of images
    labels: numpy.ndarray
        A set of image classes
    mirror_map: list or numpy.ndarray
        A mapping where 'Class i' is mirrored to imitate 'Class mirror_map[i]'
    n_new_images_max: int
        The maximum number of new images created by mirroring

    Outputs
    -------
    mirrored_images: numpy.ndarray
        A set of images with additional images
    mirrored_labels: numpy.ndarray
        A set of labels with updated count for each class

    """

    classes, classes_count = np.unique(labels, return_counts = True)

    # Number of new images that will be added to any mirrored class
    n_new_images = min(int(np.ceil(np.mean(classes_count))), n_new_images_max)

    additional_images = []
    additional_labels = []

    for class_i in classes:
        if mirror_map[class_i] != -1:

            # Find all indices mapping to the images that will be mirrored
            mirror_indices = np.where(labels == mirror_map[class_i])

            # Flip all relevant images about the mid vertical axis
            new_images = images[mirror_indices][:, :, ::-1, :]

            # Ensure that only 'n_new_images' is added
            new_images = new_images[0 : n_new_images]

            # Create new labels of 'class_i' for 'new_images'
            new_labels = np.full((len(new_images)), class_i, dtype = int)

            # Store new data
            additional_images.extend(new_images)
            additional_labels.extend(new_labels)


    augmented_images = np.concatenate((images, additional_images), axis = 0)
    augmented_labels = np.concatenate((labels, additional_labels), axis = 0)

    return augmented_images, augmented_labels

# Random transforms

def scale_image(image, scale_x, scale_y):
    """
    Scale image scene
    
    Inputs
    ----------
    image : numpy.ndarray
        Array containing a single RGB image
    scale_x,scale_y: float, float
        Scale coefficients in the x-dir and y-dir
        
    Outputs
    -------
    scaled_image: numpy.ndarray
        Scaled image, dimensions preserved
        
    """

    n_rows, n_cols, _ = image.shape

    width = int(n_cols * scale_x)
    height = int(n_rows * scale_y)

    image = cv2.resize(image, (width, height))

    image = cv2.copyMakeBorder(image, BORDER_PAD, BORDER_PAD, 
                                      BORDER_PAD, BORDER_PAD, cv2.BORDER_REPLICATE)

    n_rows_new, n_cols_new, _ = image.shape

    row_diff = round((n_rows_new - n_rows) / 2)
    col_diff = round((n_cols_new - n_cols) / 2)  

    scaled_image = image[row_diff:n_rows + row_diff, col_diff:n_cols + col_diff]

    return scaled_image

def translate_image(image, T_x, T_y):
    """
    Translate image scene
    
    Inputs
    ----------
    image : numpy.ndarray
        Array containing a single RGB image
    T_x,T_y: int, int
        Translation in the x-dir and y-dir
       
    Outputs
    -------
    translated_image: numpy.ndarray
        Translated image, dimensions preserved
        
    """

    image = cv2.copyMakeBorder(image, BORDER_PAD, BORDER_PAD, 
                                      BORDER_PAD, BORDER_PAD, cv2.BORDER_REPLICATE)

    n_rows, n_cols, _ = image.shape

    T_matrix = np.float32([[1, 0, T_x],[0, 1, T_y]])
    image = cv2.warpAffine(image, T_matrix, (n_cols, n_rows))

    translated_image = image[BORDER_PAD:n_rows - BORDER_PAD, BORDER_PAD:n_cols - BORDER_PAD]

    return translated_image

def perspective_transform(image, border_offset):
    """
    Apply perspective transform to image scene
    
    Inputs
    ----------
    image : numpy.ndarray
        Array containing a single RGB image

    Outputs
    -------
    perspective_image: numpy.ndarray
        Image with changed perspective, dimensions preserved
        
    """  
    
    image = cv2.copyMakeBorder(image, BORDER_PAD, BORDER_PAD, 
                                      BORDER_PAD, BORDER_PAD, cv2.BORDER_REPLICATE)
    n_rows, n_cols, _ = image.shape
    
    src_pts = np.float32([[border_offset, border_offset], # Top left
                          [n_cols - border_offset, border_offset], # Top right
                          [border_offset, n_rows - border_offset], # Bottom left
                          [n_cols - border_offset, n_rows - border_offset]]) # Bottom right
    dst_pts = np.float32([[0, 0],[n_cols, 0],[0, n_rows],[n_cols, n_rows]])
    
    P_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    image = cv2.warpPerspective(image, P_matrix , (n_rows, n_cols))
    
    perspective_image = image[BORDER_PAD:n_rows - BORDER_PAD, BORDER_PAD:n_cols - BORDER_PAD]
    
    return perspective_image

def rotate_image(image, angle):
    """
    Rotate image scene
    
    Inputs
    ----------
    image : numpy.ndarray
        Numpy array containing a single RGB image
    angle: int
        Angle of rotation for image in degrees
        
    Outputs
    -------
    rotated_image: numpy.ndarray
        Rotated image, dimensions preserved
        
    """
    
    image = cv2.copyMakeBorder(image, BORDER_PAD, BORDER_PAD, 
                                      BORDER_PAD, BORDER_PAD, cv2.BORDER_REPLICATE)
    n_rows, n_cols, _ = image.shape
    
    R_matrix = cv2.getRotationMatrix2D((n_rows / 2, n_cols / 2), angle, 1)
    image = cv2.warpAffine(image, R_matrix, (n_cols, n_rows))
    
    rotated_image = image[BORDER_PAD:n_rows - BORDER_PAD, BORDER_PAD:n_cols - BORDER_PAD]
    
    return rotated_image

def random_transforms(image, mask = None):
    """
    Apply one or more random transformation(s) to an image
    
    Inputs
    ----------
    image : numpy.ndarray
        Array containing a single RGB image
    mask: (None | numpy.ndarray)
        Array with shape '(1,4)' for choosing transform(s). Useful for debugging.

    Outputs
    -------
    image: numpy.ndarray
        Transformed image, dmensions preserved
        
    """ 

    if mask is None:
        mask = np.random.randint(0, 2, [1, 4])[0]

    if np.sum(mask) == 0:
        mask[np.random.randint(4)] == 1

    if mask[0] == 1:
        scale_fx = np.random.uniform(MIN_SCALE, MAX_SCALE)
        scale_fy = np.random.uniform(MIN_SCALE, MAX_SCALE)        
        image = scale_image(image, scale_fx, scale_fy)

    if mask[1] == 1:
        T_x = np.random.randint(MIN_T, MAX_T) 
        T_y = np.random.randint(MIN_T, MAX_T)
        image = translate_image(image, T_x, T_y)

    if mask[2] == 1:
        border_offset = np.random.uniform(MIN_BORDER_OFFSET, MAX_BORDER_OFFSET)
        image = perspective_transform(image, border_offset)

    if mask[3] == 1:
        angle = np.random.uniform(MIN_ANGLE, MAX_ANGLE)
        image = rotate_image(image, angle)

    return image

def augment_data_by_random_transform(images, labels):
    """
    Augment a data set by creating new samples from random transforms
    
    Inputs
    ----------
    images : numpy.ndarray
        A set of images
    labels: numpy.ndarray
        A set of image classes
        
    Outputs
    -------
    augmented_images: numpy.ndarray
        A set of images with additional samples
    augmented_labels: numpy.ndarray
        A set of classes with updated count for each class

    """
    
    classes, classes_count = np.unique(labels, return_counts = True)
    
    # Number of new images that will be added to any class
    target_count = np.max(classes_count)
    
    additional_images = []
    additional_labels = []
    
    for class_i in classes:
        
        class_count = classes_count[class_i]
        class_images_indices = np.where(labels == class_i)[0]
        
        diff_count = target_count - class_count
        if diff_count < target_count:
            
            # Pick out a subset of class_image_indices with length 'diff_count'
            indices_subset = np.random.choice(class_images_indices, 
                                              diff_count, replace = diff_count > class_count)
            
            for index in indices_subset:
                additional_images.append(random_transforms(images[index]))
                additional_labels.append(class_i)
    
    augmented_images = np.concatenate((images, additional_images), axis = 0)
    augmented_labels = np.concatenate((labels, additional_labels), axis = 0)
    
    return augmented_images, augmented_labels