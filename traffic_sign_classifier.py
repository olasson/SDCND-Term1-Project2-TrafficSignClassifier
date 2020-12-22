from code.io import data_load_pickled, data_save_pickled

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
        '--random',
        action = 'store_true',
        help = 'Can be paired with [show_images] to cause different behavior.'
    )

    args = parser.parse_args()

    # ---------- Pack arguments ---------- #

    data_paths = np.array([args.data_train, args.data_valid, args.data_test])

    flag_show_images = args.show_images
    flag_random = args.random

    # ---------- Show data ---------- #

    if flag_show_images:

        for data_path in data_paths:
            if file_exists(data_path):
                print("Loading data set:", data_path, "...")
                X, y = data_load_pickled(data_path)




main()