from code.io import data_load_pickled, data_save_pickled
from code.helpers import images_pick_subset
from code.show import show_images

from pandas import read_csv
import numpy as np
import argparse

from os.path import exists as file_exists

PATH_METADATA = 'signnames.csv'
KEY_METADATA = 'SignName'

SHOW_TRAIN = 'train'
SHOW_VALID = 'valid'
SHOW_TEST = 'test'


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
        type = str,
        nargs = '+',
        default = '',
        help = 'Specify which datasets will be visualized. Permitted values: [' + SHOW_TRAIN + ', ' + SHOW_VALID + ', ' + SHOW_TEST +']. All other values are ignored.'
    )

    parser.add_argument(
        '--show_n_images_max',
        type = int,
        nargs = '?',
        default = 25,
        help = 'The maximum number of images that can be shown.'
    )

    args = parser.parse_args()

    # ---------- Setup ---------- #

    path_train = args.data_train
    path_valid = args.data_valid
    path_test = args.data_test

    flag_show_train = SHOW_TRAIN in args.show_images
    flag_show_valid = SHOW_VALID in args.show_images
    flag_show_test = SHOW_TEST in args.show_images

    n_images_max = args.show_n_images_max

    try:
        y_metadata = read_csv(PATH_METADATA)[KEY_METADATA]
    except FileNotFoundError:
        print("Metadata not found!")
        y_metadata = None

    # ---------- Load data requested by user ---------- #
        

    if file_exists(path_train):
        print("Loading training data...")
        X_train, y_train = data_load_pickled(path_train)

    if file_exists(path_valid):
        print("Loading validation data...")
        X_valid, y_valid = data_load_pickled(path_valid)

    if file_exists(path_test):
        print("Loading testing data")
        X_test, y_test = data_load_pickled(path_test)

    # ---------- Show data ---------- #

    if flag_show_train and file_exists(path_train):
        # Too many images to show them all, pick a subset
        X_sub, y_sub, y_metadata_sub = images_pick_subset(X_train, y_train, y_metadata, n_images_max = n_images_max)
        show_images(X_sub, y_metadata_sub, title_fig_window = path_train)

    if flag_show_valid and file_exists(path_valid):
        # Too many images to show them all, pick a subset
        X_sub, y_sub, y_metadata_sub = images_pick_subset(X_valid, y_valid, y_metadata, n_images_max = n_images_max)
        show_images(X_sub, y_metadata_sub, title_fig_window = path_valid)

    if flag_show_test and file_exists(path_test):
        # Too many images to show them all, pick a subset
        X_sub, y_sub, y_metadata_sub = images_pick_subset(X_test, y_test, y_metadata, n_images_max = n_images_max)
        show_images(X_sub, y_metadata_sub, title_fig_window = path_test)






main()