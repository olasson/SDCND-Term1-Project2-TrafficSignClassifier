from code.augment import augment_data_by_mirroring, augment_data_by_random_transform
from code.process import pre_process
from code.show import show_images, show_label_distributions
from code.io import data_load_pickled, data_save_pickled
from code.helpers import images_pick_subset, model_exists
from code.models import LeNet, VGG16

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from pandas import read_csv
import numpy as np
import argparse

from os.path import exists as file_exists
from os.path import isdir as folder_exists
from os import mkdir

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
MODEL_LOSS = 'sparse_categorical_crossentropy'
MODEL_METRICS = ['accuracy']
MODEL_TRAINING_PATIENCE = 3
MODEL_TRAINING_MODE = 'max'
MODEL_TRAINING_METRIC = 'val_accuracy'



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
        '--max_epochs',
        type = int,
        nargs = '?',
        default = 50,
        help = 'The maximum number of model training epochs. The model callback can stop training much earlier.'
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
    flag_model_provided = False
    flag_model_exists = False
    flag_model_is_loaded = False
    flag_model_train = False
    flag_model_evaluate = False

    # Data
    flag_data_train_loaded = False
    flag_data_valid_loaded = False
    flag_data_test_loaded = False

    # Misc
    flag_force_save = False

    # ---------- Setup ---------- #

    # ---------- Data Setup ---------- #

    path_train = args.data_train
    path_valid = args.data_valid
    path_test = args.data_test

    if file_exists(path_train):
        flag_data_train_loaded = True

    if file_exists(path_valid):
        flag_data_valid_loaded = True

    if file_exists(path_test):
        flag_data_test_loaded = True

    # ---------- Show Setup ---------- #

    if args.show is not None:

        if len(args.show) > 0:

            flag_show_images = 'images' in args.show
            flag_show_distributions = 'dist' in args.show
            flag_show_predictions = 'predictions' in args.show

    # Set a valid order index. This might cause user input to be ignored.
    if flag_data_train_loaded and (args.dist_order == 'train'):
        order_index = 0
    elif flag_data_test_loaded and (args.dist_order == 'test'):
        order_index = 1
    elif flag_data_valid_loaded and (args.dist_order == 'valid'):
        order_index = 2
    elif (not flag_data_train_loaded) and flag_data_test_loaded:
        order_index = 1
    elif (not flag_data_train_loaded) and flag_data_valid_loaded:
        order_index = 2
    else:
        order_index = 2
    
    # ---------- Prepare Setup ---------- #

    if args.prepare_data is not None:

        flag_prepare = True

        if len(args.prepare_data) > 0:
            flag_mirroring = 'mirroring' in args.prepare_data
            flag_random_transform = 'rand_tf'in args.prepare_data

    

    # ---------- Model Setup ---------- #

    model_name = args.model_name
    if model_name:
        flag_model_provided = True

    model_path = PATH_MODEL_BASE_FOLDER + model_name + '/'

    flag_model_evaluate = args.model_evaluate

    # Model hyperparams
    batch_size = args.batch_size
    lrn_rate = args.lrn_rate
    max_epochs = args.max_epochs

    if model_exists(model_path):
        flag_model_exists = True
    else:
        # User has created a new model, train it
        flag_model_train = True

    # ---------- Misc Setup ---------- #

    if args.force_save:
        flag_force_save = True
        
    # ---------- Folder Setup ---------- #

    if not folder_exists(PATH_PREPARED_FOLDER):
        mkdir(PATH_PREPARED_FOLDER)

    if not folder_exists(PATH_MODEL_BASE_FOLDER):
        mkdir(PATH_MODEL_BASE_FOLDER)

    if not folder_exists(model_path):
        mkdir(model_path)


    # ---------- Metadata Setup ---------- #

    try:
        y_metadata = read_csv(PATH_METADATA)[KEY_METADATA]
    except FileNotFoundError:
        print("Metadata not found!")
        y_metadata = None


    # ---------- Argument Checks ---------- #

    # Try to catch argument combinations that either:
    # A) Leads to the program crashing
    # B) Causes the program to do nothing, which can be confusing.
    

    # User is trying to show images, but no data is loaded
    if not (flag_data_train_loaded or flag_data_valid_loaded or flag_data_test_loaded):
        print("ERROR: No data is loaded, nothing can be done without some data!")
        return


    if flag_show_distributions:

        # User has selected an order index value that corresponds to an empty label set
        if ((order_index == 0 and (not flag_data_train_loaded)) or 
            (order_index == 1 and (not flag_data_test_loaded)) or 
            (order_index == 2 and (not flag_data_valid_loaded))):
            print("ERROR: main(): The selected order distribution is 'None'!")
            return

    if not flag_model_provided:

        # User has requested evaluation or prediction, but provided no model
        if flag_model_evaluate or flag_show_predictions:
            print("ERROR: main(): You must provide a model in order to do evaluation and/or prediction!")
            return

    if flag_model_provided:

        # User has provided a model and requested training, but no training or validation data is loaded
        if flag_model_train and not (flag_data_train_loaded or flag_data_valid_loaded):
            print("ERROR: main(): You are trying to train your model, but no training and validation data is loaded!")
            return

    if not flag_model_exists:

        # User is trying to evalute a model that does not exist, and no training is requested either
        if flag_model_evaluate and not (flag_model_train or flag_force_save):
            print("ERROR: main(): Your are trying to evaluate a model that does not exist!")
            return

    if flag_model_exists:

        # User is trying to evaluate an existing model, but no testing data is provided.
        if flag_model_evaluate and (not flag_data_test_loaded):
            print("ERROR: main(): You are trying to evalutate your model, but no testing data is provided!")
            return


    # ---------- Load data requested by user ---------- #
        
    if file_exists(path_train):
        print("Loading training data...")
        X_train, y_train = data_load_pickled(path_train)
    else:
        X_train, y_train = None, None

    if file_exists(path_valid):
        print("Loading validation data...")
        X_valid, y_valid = data_load_pickled(path_valid)
    else:
        X_valid, y_valid = None, None

    if file_exists(path_test):
        print("Loading testing data...")
        X_test, y_test = data_load_pickled(path_test)
    else:
        X_test, y_test = None, None


    # ---------- Visualize data ---------- #

    # Images

    if flag_show_images:

        if flag_data_train_loaded:
            # Too many images to show them all, pick a subset
            X_sub, _, y_metadata_sub = images_pick_subset(X_train, y_train, y_metadata, n_images_max = N_IMAGES_MAX)
            show_images(X_sub, y_metadata_sub, title_fig_window = path_train)

        if flag_data_valid_loaded:
            # Too many images to show them all, pick a subset
            X_sub, _, y_metadata_sub = images_pick_subset(X_valid, y_valid, y_metadata, n_images_max = N_IMAGES_MAX)
            show_images(X_sub, y_metadata_sub, title_fig_window = path_valid)

        if flag_data_test_loaded:
            # Too many images to show them all, pick a subset
            X_sub, _, y_metadata_sub = images_pick_subset(X_test, y_test, y_metadata, n_images_max = N_IMAGES_MAX)
            show_images(X_sub, y_metadata_sub, title_fig_window = path_test)

    # Distributions

    if flag_show_distributions:

        title = ''
        if flag_data_train_loaded:
            title += "| Training set (Blue) | "
        if flag_data_test_loaded:
            title += "| Testing set (Orange) |"
        if flag_data_valid_loaded:
            title += "| Validation set (Green) |"

        show_label_distributions([y_train, y_test, y_valid], y_metadata, title = title, order_index = order_index, colors = COLORS)


    # ---------- Prepare data ---------- #

    if flag_prepare:

        # ---------- Prepare Training data ---------- #

        if flag_data_train_loaded:

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

        if flag_data_valid_loaded:

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

        if flag_data_test_loaded:

            if (not file_exists(PATH_PREPARED_TEST)) or flag_force_save:

                print("Pre-processing validation data...")
                X_test = pre_process(X_test)

                print("Saving testing data...")
                data_save_pickled(PATH_PREPARED_TEST, X_test, y_test)

            else:
                print("Testing data exists, skipping!")

        else:
            print("Testing data not provided, skipping preparation!")

    if flag_model_provided:

        model = VGG16()
        optimizer = Adam(learning_rate = lrn_rate)

        # flag_force_save: User has the option to force a new training session, even if the model exists 
        if (flag_model_train and flag_data_train_loaded and flag_data_valid_loaded) or flag_force_save:
            
            model.compile(optimizer = optimizer, loss = MODEL_LOSS, metrics = MODEL_METRICS)

            early_stopping = EarlyStopping(monitor = MODEL_TRAINING_METRIC, 
                                           patience = MODEL_TRAINING_PATIENCE, min_delta = lrn_rate, 
                                           mode = MODEL_TRAINING_MODE, restore_best_weights = True)
            model.fit(X_train, y_train, batch_size = batch_size, epochs = max_epochs, 
                        validation_data = (X_valid, y_valid), callbacks = [early_stopping])

            print("Saving", model_name, "...")
            model.save_weights(model_path)

            flag_model_is_loaded = True

        if (flag_model_evaluate and flag_data_test_loaded):

            if (not flag_model_is_loaded):

                print("Loading", model_name, "...")
                model.load_weights(model_path).expect_partial()
                model.compile(optimizer = optimizer, loss = MODEL_LOSS, metrics = MODEL_METRICS)

                flag_model_is_loaded = True

            print("Evaluating", model_name, "...")
            model.evaluate(X_test, y_test, batch_size = batch_size) 
            


main()