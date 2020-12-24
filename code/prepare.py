import numpy as np
import cv2

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