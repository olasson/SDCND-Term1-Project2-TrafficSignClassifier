import argparse
import os


def main():

    # ---------- Command line arguments ---------- #

    parser = argparse.ArgumentParser(description = 'Detect Lane Lines')

    parser.add_argument(
        '--show_images',
        type = str,
        nargs = '?',
        default= '',
        help = 'Show a set of images.'
    )

    parser.add_argument(
        '--random',
        type = bool,
        default = False,
        help = 'Can be paired with [--show_images] to cause different behavior.'
    )

    args = parser.parse_args()

    if args.show_images:
        print("Test")

        if args.random:
            print("Test2")


main()