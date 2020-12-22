from code.io import data_load_pickled, data_save_pickled
from code.helpers import images_pick_subset
from code.show import show_images

from pandas import read_csv
import numpy as np
import argparse

from os.path import exists as file_exists


TRAIN = 0
VALID = 1
TEST = 2


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
        help = 'Show a set of images.'
    )

    parser.add_argument(
        '--show_images_max',
        type = int,
        default = 25,
        help = 'Maximum number of images displayed.'
    )

    args = parser.parse_args()

    # ---------- Setup ---------- #

    data_paths = np.array([args.data_train, args.data_valid, args.data_test])

    flag_show_images = args.show_images

    n_images_max = args.show_images_max

    try:
        y_metadata = read_csv('signnames.csv')['SignName']
    except FileNotFoundError:
        y_metadata = None
        print("No metadata file found!")

    # ---------- Show data ---------- #

    if flag_show_images:

        for data_path in data_paths:
            if file_exists(data_path):
                
                X, y = data_load_pickled(data_path)

                # Pick out a subset of images if 'len(X) > n_images_max'
                X_sub, y_sub, y_metadata_sub = images_pick_subset(X, y, y_metadata, n_images_max = n_images_max)

                show_images(X_sub, y_sub, y_metadata_sub, title_window = data_path)




main()