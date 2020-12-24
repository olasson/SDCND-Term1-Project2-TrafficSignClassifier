from code.prepare import augment_data_by_mirroring
from code.show import show_images, show_label_distributions
from code.io import data_load_pickled, data_save_pickled
from code.helpers import images_pick_subset


from pandas import read_csv
import numpy as np
import argparse

from os.path import exists as file_exists


# Metadata info
PATH_METADATA = 'signnames.csv'
KEY_METADATA = 'SignName'

# Fixed names for prepared data
PATH_PREPARED_FOLDER = './data/'
PATH_PREPARED_TRAIN = PATH_PREPARED_FOLDER + 'prepared_train.p'
PATH_PREPARED_VALID = PATH_PREPARED_FOLDER + 'prepared_valid.p'
PATH_PREPARED_TEST = PATH_PREPARED_FOLDER + 'prepared_test.p'

# Permitted options for --prepare command
DATA_PREPARATION_OPTIONS = ['mirroring', 'rand_tf', 'hist_eq']

# Mapping where "Class i" is mirrored to imitate "Class MIRROR_MAP[i]"
# Used by the --prepare command
MIRROR_MAP = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
              -1, 11, 12, 13, -1, 15, -1, 17, 18, 20,
              19, -1, 22, -1, -1, -1, 26, -1, -1, -1,
              30, -1, -1, 34, 33, 35, 37, 36, 39, 38,
              -1, -1, -1]

def main():

    # ---------- Command line arguments ---------- #

    parser = argparse.ArgumentParser(description = 'Traffic Sign Classifier')

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
        action = 'store_true',
        help = 'Shows images from all datasets provided.'
    )

    parser.add_argument(
        '--n_images_max',
        type = int,
        nargs = '?',
        default = 25,
        help = 'The maximum number of images that can be shown.'
    )

    parser.add_argument(
        '--show_distributions',
        action = 'store_true',
        help = 'Shows class distributions from all datasets provided.'
    )

    parser.add_argument(
        '--title',
        type = str,
        nargs = '?',
        default = '',
        help = 'Title for distribution plot'
    )

    parser.add_argument(
        '--order_index',
        type = int,
        nargs = '?',
        default = 0,
        help = 'Determines which distribution will be used to set the class order on the y-axis.'
    )

    parser.add_argument(
        '--prepare',
        type = str,
        nargs = '*',
        help = 'Initiates data preparation (augmentation + pre-processing). If no flags are provided, all steps are performed.'
    )


    args = parser.parse_args()

    # ---------- Setup ---------- #

    path_train = args.data_train
    path_valid = args.data_valid
    path_test = args.data_test

    flag_show_images = args.show_images
    flag_show_distributions = args.show_distributions

    if args.prepare is not None:

        flag_prepare = True

        if len(args.prepare) > 0:
            flag_mirroring = DATA_PREPARATION_OPTIONS[0] in args.prepare
            flag_random_transform = DATA_PREPARATION_OPTIONS[1] in args.prepare
            flag_histogram_equalization = DATA_PREPARATION_OPTIONS[2] in args.prepare
        else: 
            flag_mirroring = True
            flag_random_transform = True
            flag_histogram_equalization = True

    else:
        flag_prepare = False


    n_images_max = args.n_images_max

    order_index = args.order_index

    title_distributions = args.title

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
        print("Loading testing data")
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
        show_label_distributions([y_train, y_test, y_valid], y_metadata, title = title_distributions, order_index = order_index)


    # ---------- Prepare data ---------- #
    if flag_prepare:

    # Prepare training data

        if flag_mirroring:
            print("MIRRORING")

        if flag_random_transform:
            print("RAND TRANSFORMS")


    # Pre-processing

        if flag_histogram_equalization:
            print("HIST EQ")





main()