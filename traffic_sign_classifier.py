from code.augment import augment_data_by_mirroring, augment_data_by_random_transform
from code.process import histogram_equalization, grayscale, normalize_images
from code.show import show_images, show_label_distributions
from code.io import data_load_pickled, data_save_pickled
from code.helpers import images_pick_subset #, distributions_title
from code.models import LeNet, VGG16

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from pandas import read_csv
import numpy as np
import argparse

from os.path import exists as file_exists
from os.path import isdir as folder_exists
from os import mkdir, listdir

def model_exists(model_path):
    return (not len(listdir(model_path)) == 0)

# General constants
N_IMAGES_MAX = 25

# Colors for distribution plot
COLORS = ['tab:blue', 'tab:orange', 'tab:green']

# Metadata info
PATH_METADATA = 'signnames.csv'
KEY_METADATA = 'SignName'

# Fixed names for prepared data
PATH_PREPARED_FOLDER = './data/'
PATH_PREPARED_TRAIN = PATH_PREPARED_FOLDER + 'prepared_train.p'
PATH_PREPARED_VALID = PATH_PREPARED_FOLDER + 'prepared_valid.p'
PATH_PREPARED_TEST = PATH_PREPARED_FOLDER + 'prepared_test.p'

# Model
PATH_MODEL_FOLDER = './models/'
DECAY_STEPS = 10000
DECAY_RATE = 0.0
MODEL_LOSS = 'sparse_categorical_crossentropy'
MODEL_METRICS = ['accuracy']


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

    # Show

    parser.add_argument(
        '--show',
        default = 'ignore_this',
        type = str,
        nargs = '*',
        choices = ['images', 'dist', 'predictions'],
        help = 'Visualize images, distributions or model predictions.'
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

    # Data Preparation

    parser.add_argument(
        '--prepare',
        default = 'ignore_this',
        type = str,
        nargs = '*',
        choices = ['mirroring', 'rand_tf'],
        help = 'Prepares data for use by a model. Optional: provide augmentation options for the training set.'
    )

    parser.add_argument(
        '--force_save',
        type = bool,
        nargs = '?',
        default = False,
        help = 'If true, existing prepared data will be overwritten.'
    )

    # Models

    parser.add_argument(
        '--model_name',
        type = str,
        nargs = '?',
        default = '',
        help = 'Name of model.'
    )

    parser.add_argument(
        '--model_type',
        default = 'VGG16',
        const = 'VGG16',
        type = str,
        nargs = '?',
        choices = ['VGG16', 'LeNet'],
        help = 'Choose model architecture'
    )

    parser.add_argument(
        '--evaluate',
        action = 'store_true',
        help = 'Evaluates the model on the entire (.p) test set.'
    )

    parser.add_argument(
        '--batch_size',
        type = int,
        nargs = '?',
        default = 64,
        help = 'Model learning rate.'
    )

    parser.add_argument(
        '--lrn_rate',
        type = int,
        nargs = '?',
        default = 0.001,
        help = 'Model learning rate.'
    )

    parser.add_argument(
        '--epochs',
        type = int,
        nargs = '?',
        default = 50,
        help = 'Model training epochs.'
    )

    args = parser.parse_args()
    
    # ---------- Setup ---------- #

    # Paths

    path_train = args.data_train
    path_valid = args.data_valid
    path_test = args.data_test

    # Show

    flag_show_images = False
    flag_show_distributions = False
    flag_show_predictions = False
    if args.show is not None:
        if len(args.show) > 0:
            flag_show_images = 'images' in args.show
            flag_show_distributions = 'dist' in args.show
            flag_show_predictions = 'predictions' in args.show

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

    flag_force_save = args.force_save

    # Models

    flag_train_model = False
    flag_evaluate_model = False

    if not folder_exists(PATH_MODEL_FOLDER):
        mkdir(PATH_MODEL_FOLDER)

    model_name = args.model_name
    model_path = PATH_MODEL_FOLDER + model_name + '/'

    if not folder_exists(model_path):
        mkdir(model_path)

    # User has created a new model, train it
    if not model_exists(model_path):
        flag_train_model = True

    # Hyperparams
    batch_size = args.batch_size
    lrn_rate = args.lrn_rate
    epochs = args.epochs



    if model_name:
        if args.model_type == 'VGG16':
            model = VGG16()
            print("Model type VGG16 chosen!")
        elif args.model_type == 'LeNet':
            model = LeNet()
            print("Model type LeNet chosen!")

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                                initial_learning_rate = lrn_rate,
                                decay_steps = DECAY_STEPS,
                                decay_rate = DECAY_RATE)
        optimizer = keras.optimizers.Adam(learning_rate = lrn_rate)


    # Metadata

    try:
        y_metadata = read_csv(PATH_METADATA)[KEY_METADATA]
    except FileNotFoundError:
        print("Metadata not found!")
        y_metadata = None

    if not folder_exists(PATH_PREPARED_FOLDER):
        mkdir(PATH_PREPARED_FOLDER)

    # ---------- Argument Checks ---------- #



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

    #print(distributions_title(y_train, y_test, y_valid))
    #return

    # ---------- Visualize data ---------- #

    # Images

    if flag_show_images:

        if (X_train is not None) and (y_train is not None):
            # Too many images to show them all, pick a subset
            X_sub, _, y_metadata_sub = images_pick_subset(X_train, y_train, y_metadata, n_images_max = N_IMAGES_MAX)
            show_images(X_sub, y_metadata_sub, title_fig_window = path_train)

        if (X_valid is not None) and (y_valid is not None):
            # Too many images to show them all, pick a subset
            X_sub, _, y_metadata_sub = images_pick_subset(X_valid, y_valid, y_metadata, n_images_max = N_IMAGES_MAX)
            show_images(X_sub, y_metadata_sub, title_fig_window = path_valid)

        if (X_test is not None) and (y_test is not None):
            # Too many images to show them all, pick a subset
            X_sub, _, y_metadata_sub = images_pick_subset(X_test, y_test, y_metadata, n_images_max = N_IMAGES_MAX)
            show_images(X_sub, y_metadata_sub, title_fig_window = path_test)

    # Distributions

    if flag_show_distributions:

        title = ''
        if y_train is not None:
            title += "| Training set (Blue) | "
        if y_test is not None:
            title += "| Testing set (Orange) |"
        if y_valid is not None:
            title += "| Validation set (Green) |"

        show_label_distributions([y_train, y_test, y_valid], y_metadata, title = title, order_index = order_index, colors = COLORS)


    # ---------- Prepare data ---------- #

    if flag_prepare:

        # ---------- Prepare Training data ---------- #

        if (X_train is not None) and (y_train is not None):

            if (not file_exists(PATH_PREPARED_TRAIN)) or flag_force_save:

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

                print("Saving training data...")

                data_save_pickled(PATH_PREPARED_TRAIN, X_train, y_train)

            else:
                print("Training data exists, skipping!")

        else:
            print("Training data not provided, skipping preparation!")

        # ---------- Prepare Validation data ---------- #

        if (X_valid is not None) and (y_valid is not None):

            if (not file_exists(PATH_PREPARED_VALID)) or flag_force_save:

                print("Pre-processing validation data...")

                X_valid = histogram_equalization(X_valid)

                X_valid = grayscale(X_valid)

                X_valid = normalize_images(X_valid, A_NORM, B_NORM, IMAGE_MIN, IMAGE_MAX)

                X_valid = X_valid[..., np.newaxis]

                print("Saving validation data...")

                data_save_pickled(PATH_PREPARED_VALID, X_valid, y_valid)

            else:
                print("Validation data exists, skipping!")

        else:
            print("Validation data not provided, skipping preparation!")

        # ---------- Prepare Testing data ---------- #

        if (X_test is not None) and (y_test is not None):

            if (not file_exists(PATH_PREPARED_TEST)) or flag_force_save:

                print("Pre-processing validation data...")

                X_test = histogram_equalization(X_test)

                X_test = grayscale(X_test)

                X_test = normalize_images(X_test, A_NORM, B_NORM, IMAGE_MIN, IMAGE_MAX)

                X_test = X_test[..., np.newaxis]

                print("Saving testing data...")

                data_save_pickled(PATH_PREPARED_TEST, X_test, y_test)

            else:
                print("Testing data exists, skipping!")

        else:
            print("Testing data not provided, skipping preparation!")

    if model_name:

        if flag_train_model:
            model.compile(optimizer = optimizer, loss = MODEL_LOSS, metrics = MODEL_METRICS)
            early_stopping = EarlyStopping(monitor = 'val_accuracy', 
                                           patience = 3, min_delta = 0.001, 
                                           mode = 'max', restore_best_weights = True)
            model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, 
                        validation_data = (X_valid, y_valid), callbacks = [early_stopping])


            print("MODEL")

        flag_evaluate_model = False
        if flag_evaluate_model:
            pass


main()