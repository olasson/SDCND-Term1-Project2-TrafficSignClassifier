from code.augment import augment_data_by_mirroring, augment_data_by_random_transform
from code.process import histogram_equalization, grayscale, normalize_images
from code.show import show_images, show_label_distributions
from code.io import data_load_pickled, data_save_pickled
from code.helpers import images_pick_subset


from pandas import read_csv
import numpy as np
import argparse

from os.path import exists as file_exists
from os.path import isdir as folder_exists
from os import mkdir


# Metadata info
PATH_METADATA = 'signnames.csv'
KEY_METADATA = 'SignName'

# Fixed names for prepared data
PATH_PREPARED_FOLDER = './data/'
PATH_PREPARED_TRAIN = PATH_PREPARED_FOLDER + 'prepared_train.p'
PATH_PREPARED_VALID = PATH_PREPARED_FOLDER + 'prepared_valid.p'
PATH_PREPARED_TEST = PATH_PREPARED_FOLDER + 'prepared_test.p'

# Mapping where "Class i" is mirrored to imitate "Class MIRROR_MAP[i]"
# Used by the --prepare command
MIRROR_MAP = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
              -1, 11, 12, 13, -1, 15, -1, 17, 18, 20,
              19, -1, 22, -1, -1, -1, 26, -1, -1, -1,
              30, -1, -1, 34, 33, 35, 37, 36, 39, 38,
              -1, -1, -1]

# Constants for normalization
A_NORM = 0
B_NORM = 1
IMAGE_MIN = 0
IMAGE_MAX = 255

def main():

    # ---------- Command line arguments ---------- #

    parser = argparse.ArgumentParser(description = 'Traffic Sign Classifier')
    sub_commands = parser.add_subparsers(title = 'sub-commands')

    parser.add_argument(
        '--data_train',
        type = str,
        nargs = '?',
        default = '',
        help = 'Path to a pickled (.p) training set.'
    )

    parser.add_argument(
        '--data_valid',
        type = str,
        nargs = '?',
        default = '',
        help = 'Path to a pickled (.p) validation set.'
    )

    parser.add_argument(
        '--data_test',
        type = str,
        nargs = '?',
        default = '',
        help = 'Path to a pickled (.p) testing set.'
    )

    parser.add_argument(
        '--show_images',
        type = int,
        nargs='*',
        help = 'Shows images from all datasets provided. Optional: provide an integer to specify the number of images shown.'
    )

    parser.add_argument(
        '--show_dist',
        type = str,
        nargs='*',
        help = 'Shows class distributions for all datasets provided. Optional: provide a single title for the plot.'
    )

    parser.add_argument(
        '--dist_order',
        default = 'train',
        const = 'train',
        type = str,
        nargs = '?',
        choices = ['train', 'test', 'valid'],
        help = 'Determines which distribution will be used to set the class order on the y-axis.'
    )

    parser.add_argument(
        '--prepare',
        default = 'ignore',
        type = str,
        nargs = '*',
        choices = ['mirroring', 'rand_tf'],
        help = 'Prepares data for use by a model. Optional: provide augmentation options for the training set.'
    )


    args = parser.parse_args()
    
    # ---------- Setup ---------- #

    # Paths

    path_train = args.data_train
    path_valid = args.data_valid
    path_test = args.data_test

    # Show_images

    flag_show_images = False
    n_images_max = 25
    if args.show_images is not None:

        flag_show_images = True

        if len(args.show_images) > 1 or min(args.show_images) < 1:
            print("ERROR: main(): --show_images: Provide a single positive integer! Your input:", args.show_images)
            return

        if len(args.show_images) > 0:  
            n_images_max = args.show_images[0]


    # Show_distribution

    flag_show_distributions = False
    dist_title = None
    if args.show_dist is not None:

        flag_show_distributions = True

        if len(args.show_dist) > 1:
            print("ERROR: main(): --show_dist: Provide a single string as title! Your input:", args.show_dist)
            return

        if len(args.show_dist) > 0:  
            dist_title = args.show_dist[0]


    dist_order = args.dist_order
    if dist_order == 'train':
        order_index = 0
    elif dist_order == 'test':
        order_index = 1
    else:
        order_index = 2
    

    # Prepare

    flag_prepare = False
    flag_mirroring = False
    flag_random_transform = False
    if args.prepare is not None:

        flag_prepare = True

        if len(args.prepare) > 0:
            flag_mirroring = 'mirroring' in args.prepare
            flag_random_transform = 'rand_tf'in args.prepare


    # Metadata

    try:
        y_metadata = read_csv(PATH_METADATA)[KEY_METADATA]
    except FileNotFoundError:
        print("Metadata not found!")
        y_metadata = None


    # ---------- Load data requested by user ---------- #
        
    if file_exists(path_train):
        print("Loading training data...")
        X_train, y_train = data_load_pickled(path_train)
    else:
        X_train = None
        y_train = None

    if file_exists(path_valid):
        print("Loading validation data...")
        X_valid, y_valid = data_load_pickled(path_valid)
    else:
        X_valid = None
        y_valid = None

    if file_exists(path_test):
        print("Loading testing data...")
        X_test, y_test = data_load_pickled(path_test)
    else:
        X_test = None
        y_test = None

    # ---------- Visualize data ---------- #

    # Images

    if flag_show_images:

        if file_exists(path_train):
            # Too many images to show them all, pick a subset
            X_sub, y_sub, y_metadata_sub = images_pick_subset(X_train, y_train, y_metadata, n_images_max = n_images_max)
            show_images(X_sub, y_metadata_sub, title_fig_window = path_train)

        if file_exists(path_valid):
            # Too many images to show them all, pick a subset
            X_sub, y_sub, y_metadata_sub = images_pick_subset(X_valid, y_valid, y_metadata, n_images_max = n_images_max)
            show_images(X_sub, y_metadata_sub, title_fig_window = path_valid)

        if file_exists(path_test):
            # Too many images to show them all, pick a subset
            X_sub, y_sub, y_metadata_sub = images_pick_subset(X_test, y_test, y_metadata, n_images_max = n_images_max)
            show_images(X_sub, y_metadata_sub, title_fig_window = path_test)

    # Distributions

    if flag_show_distributions:
        show_label_distributions([y_train, y_test, y_valid], y_metadata, title = dist_title, order_index = order_index)


    # ---------- Prepare data ---------- #
    if flag_prepare:

    # Prepare training data

        if (X_train is not None) and (y_train is not None):

            # Agumentation

            if flag_mirroring:
                print("Mirroring training data...")
                X_train, y_train = augment_data_by_mirroring(X_train, y_train, MIRROR_MAP)

            if flag_random_transform:
                print("Applying random transforms to training data...")
                X_train, y_train = augment_data_by_random_transform(X_train, y_train)

            # Pre-processing

            print("Pre-processing training data...")

            X_train = histogram_equalization(X_train)

            X_train = grayscale(X_train)

            X_train = normalize_images(X_train, A_NORM, B_NORM, IMAGE_MIN, IMAGE_MAX)

            X_train = X_train[..., np.newaxis]

        else:
            print("Training data not provided, skipping preparation!")

        if (X_valid is not None) and (y_valid is not None):

            print("Pre-processing validation data...")

            X_valid = histogram_equalization(X_valid)

            X_valid = grayscale(X_valid)

            X_valid = normalize_images(X_valid, A_NORM, B_NORM, IMAGE_MIN, IMAGE_MAX)

            X_valid = X_valid[..., np.newaxis]

        else:
            print("Validation data not provided, skipping preparation!")

        if (X_test is not None) and (y_test is not None):

            print("Pre-processing validation data...")

            X_test = histogram_equalization(X_test)

            X_test = grayscale(X_test)

            X_test = normalize_images(X_test, A_NORM, B_NORM, IMAGE_MIN, IMAGE_MAX)

            X_test = X_test[..., np.newaxis]
            
        else:
            print("Testing data not provided, skipping preparation!")


main()