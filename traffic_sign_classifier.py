from code.io import data_load_pickled, data_save_pickled


import argparse
import os


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


main()