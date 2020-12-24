import numpy as np
import cv2

# ---------- Constand for random transforms ---------- #
BORDER_PAD = 10


# ---------- Data augmentation ---------- #

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
    border_pad: int
        Padding used to ensure image size preservation
        
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