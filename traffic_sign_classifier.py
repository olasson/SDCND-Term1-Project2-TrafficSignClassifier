from code.augment import augment_data_by_mirroring, augment_data_by_random_transform
from code.process import pre_process
from code.show import show_images, show_label_distributions
from code.io import data_load_pickled, data_save_pickled
from code.helpers import images_pick_subset
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
PATH_MODEL_BASE_FOLDER = './models/'
#DECAY_STEPS = 10000
#DECAY_RATE = 0.0
MODEL_LOSS = 'sparse_categorical_crossentropy'
MODEL_METRICS = ['accuracy']


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
        '--prepare_data',
        default = None,
        type = str,
        nargs = '*',
        choices = ['mirroring', 'rand_tf'],
        help = 'Prepares data for use by a model. Optional: provide augmentation options for the training set.'
    )

    parser.add_argument(
        '--force_save',
        action = 'store_true',
        help = 'If true, existing prepared data and/or models will be overwritten. Use with care!'
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
        '--model_evaluate',
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
    
    # ---------- Init Flags ---------- #

    # Show
    flag_show_images = False
    flag_show_distributions = False
    flag_show_predictions = False

    # Prepare
    flag_prepare = False
    flag_mirroring = False
    flag_random_transform = False

    # Model
    flag_train_model = False
    flag_evaluate_model = False

    # Data
    flag_train_data_loaded = False
    flag_valid_data_loaded = False
    flag_test_data_loaded = False

    # ---------- Setup ---------- #

    # Paths

    path_train = args.data_train
    path_valid = args.data_valid
    path_test = args.data_test

    # Show
    if args.show is not None:

        if len(args.show) > 0:

            flag_show_images = 'images' in args.show
            flag_show_distributions = 'dist' in args.show
            flag_show_predictions = 'predictions' in args.show

    if args.dist_order == 'train':
        order_index = 0
    elif args.dist_order == 'test':
        order_index = 1
    else:
        order_index = 2
    

    # Prepare
    if args.prepare_data is not None:

        flag_prepare = True

        if len(args.prepare_data) > 0:
            flag_mirroring = 'mirroring' in args.prepare_data
            flag_random_transform = 'rand_tf'in args.prepare_data

    flag_force_save = args.force_save


    model_name = args.model_name
    model_path = PATH_MODEL_BASE_FOLDER + model_name + '/'

    flag_evaluate_model = args.model_evaluate

    # Model hyperparams
    batch_size = args.batch_size
    lrn_rate = args.lrn_rate
    epochs = args.epochs

    # User has created a new model, train it
    if not model_exists(model_path):
        flag_train_model = True

    if not folder_exists(PATH_PREPARED_FOLDER):
        mkdir(PATH_PREPARED_FOLDER)

    if not folder_exists(PATH_MODEL_BASE_FOLDER):
        mkdir(PATH_MODEL_BASE_FOLDER)

    if not folder_exists(model_path):
        mkdir(model_path)


    # Load Metadata

    try:
        y_metadata = read_csv(PATH_METADATA)[KEY_METADATA]
    except FileNotFoundError:
        print("Metadata not found!")
        y_metadata = None

    # ---------- Load data requested by user ---------- #
        
    if file_exists(path_train):
        print("Loading training data...")
        X_train, y_train = data_load_pickled(path_train)
        flag_train_data_loaded = True
    else:
        X_train, y_train = None, None

    if file_exists(path_valid):
        print("Loading validation data...")
        X_valid, y_valid = data_load_pickled(path_valid)
        flag_valid_data_loaded = True
    else:
        X_valid, y_valid = None, None


    if file_exists(path_test):
        print("Loading testing data...")
        X_test, y_test = data_load_pickled(path_test)
        flag_test_data_loaded = True
    else:
        X_test, y_test = None, None

    # ---------- Argument Checks ---------- #
    if flag_show_distributions:
        # User has selected a order index value that corresponds to an empty label set
        if (order_index == 0 and y_train is None) or (order_index == 1 and y_test is None) or (order_index == 0 and y_valid is None):
            print("ERROR: main(): --dist_order: The selected order distribution is 'None'!")
            return

    if model_name:
        # User is trying to evaluate a model that does not exist, and no training is requested either
        if flag_evaluate_model and (not model_exists(model_path)) and (not flag_train_model) and (not flag_force_save):
            print("ERROR: main(): --model_evaluate: You are trying to evaulate a model that does not exist!")
            return

        # User is trying to evaluate their model, but the needed data is not loaded
        if flag_evaluate_model and (not flag_test_data_loaded):
            print("ERROR: main(): --model_evaluate: You are trying to evaluate your model, but the testing data is not loaded!")
            return

        # User is trying to train their model, but the needed data is not loaded
        if flag_train_model and not (flag_train_data_loaded or flag_valid_data_loaded):
            print("ERROR: main() --model_name: You are trying to train your model, but training and validation data is not loaded!")
            return


    # ---------- Visualize data ---------- #

    # Images

    if flag_show_images:

        if flag_train_data_loaded:
            # Too many images to show them all, pick a subset
            X_sub, _, y_metadata_sub = images_pick_subset(X_train, y_train, y_metadata, n_images_max = N_IMAGES_MAX)
            show_images(X_sub, y_metadata_sub, title_fig_window = path_train)

        if flag_valid_data_loaded:
            # Too many images to show them all, pick a subset
            X_sub, _, y_metadata_sub = images_pick_subset(X_valid, y_valid, y_metadata, n_images_max = N_IMAGES_MAX)
            show_images(X_sub, y_metadata_sub, title_fig_window = path_valid)

        if flag_test_data_loaded:
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

        if flag_train_data_loaded:

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
                X_train = pre_process(X_train)

                print("Saving training data...")
                data_save_pickled(PATH_PREPARED_TRAIN, X_train, y_train)

            else:
                print("Training data exists, skipping!")

        else:
            print("Training data not provided, skipping preparation!")

        # ---------- Prepare Validation data ---------- #

        if flag_valid_data_loaded:

            if (not file_exists(PATH_PREPARED_VALID)) or flag_force_save:

                print("Pre-processing validation data...")
                X_valid = pre_process(X_valid)

                print("Saving validation data...")
                data_save_pickled(PATH_PREPARED_VALID, X_valid, y_valid)

            else:
                print("Validation data exists, skipping!")

        else:
            print("Validation data not provided, skipping preparation!")

        # ---------- Prepare Testing data ---------- #

        if flag_test_data_loaded:

            if (not file_exists(PATH_PREPARED_TEST)) or flag_force_save:

                print("Pre-processing validation data...")
                X_test = pre_process(X_test)

                print("Saving testing data...")
                data_save_pickled(PATH_PREPARED_TEST, X_test, y_test)

            else:
                print("Testing data exists, skipping!")

        else:
            print("Testing data not provided, skipping preparation!")

    if model_name:

        flag_model_is_loaded = False

        model = VGG16()
        optimizer = keras.optimizers.Adam(learning_rate = lrn_rate)

        # User has the option to force a new training session, even if the model exists 
        if (flag_train_model and flag_train_data_loaded and flag_valid_data_loaded) or flag_force_save:
            
            model.compile(optimizer = optimizer, loss = MODEL_LOSS, metrics = MODEL_METRICS)

            early_stopping = EarlyStopping(monitor = 'val_accuracy', 
                                           patience = 3, min_delta = 0.001, 
                                           mode = 'max', restore_best_weights = True)
            model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, 
                        validation_data = (X_valid, y_valid), callbacks = [early_stopping])

            print("Saving", model_name, "...")
            model.save_weights(model_path)

            flag_model_is_loaded = True

        if (flag_evaluate_model and flag_test_data_loaded):

            if (not flag_model_is_loaded):

                print("Loading", model_name, "...")
                model.load_weights(model_path).expect_partial()
                model.compile(optimizer = optimizer, loss = MODEL_LOSS, metrics = MODEL_METRICS)

            print("Evaluating", model_name, "...")
            model.evaluate(X_test, y_test, batch_size = batch_size) 
            


main()