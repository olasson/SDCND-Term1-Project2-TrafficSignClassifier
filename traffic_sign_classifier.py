"""
This is the main project file for the Traffic Sign Classifier Project.
It contains an implementation of a (very) basic command line tool
"""

# ---------- Custom Imports ---------- #

from code.helpers import images_pick_subset, predictions_create_titles, folder_is_empty, dist_is_uniform, web_data_file_names_are_valid
from code.augment import augment_data_by_mirroring, augment_data_by_random_transform
from code.io import data_load_pickled, data_save_pickled, data_load_web
from code.show import show_images, show_label_distributions, plot_model_history
from code.models import LeNet, VGG16
from code.process import pre_process

# ---------- General Imports ---------- #

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

from pandas import read_csv
import numpy as np
import argparse

from os.path import exists as file_exists
from os.path import isdir as folder_exists
from os import mkdir, listdir

# ---------- Config ---------- #

# General constants
N_IMAGES_MAX = 25
N_PREDICTIONS_MAX = 15

# Metadata info
PATH_METADATA = 'signnames.csv'
KEY_METADATA = 'SignName'

# Raw data
PATH_RAW_FOLDER = './data/'
PATH_RAW_TRAIN = PATH_RAW_FOLDER + 'train.p'
PATH_RAW_VALID = PATH_RAW_FOLDER + 'valid.p'
PATH_RAW_TEST = PATH_RAW_FOLDER + 'test.p'

# Prepared data
PATH_PREPARED_FOLDER = './data/'
PATH_PREPARED_TRAIN = PATH_PREPARED_FOLDER + 'prepared_train.p'
PATH_PREPARED_VALID = PATH_PREPARED_FOLDER + 'prepared_valid.p'
PATH_PREPARED_TEST = PATH_PREPARED_FOLDER + 'prepared_test.p'

# Model
PATH_MODEL_FOLDER = './models/'
MODEL_LOSS = 'sparse_categorical_crossentropy'
MODEL_METRICS = ['accuracy']
MODEL_TRAINING_PATIENCE = 5
MODEL_TRAINING_MODE = 'max'
MODEL_TRAINING_METRIC = 'val_accuracy'
MODEL_TRAINING_MIN_DELTA = 0.001


# Mapping where "Class i" is mirrored to imitate "Class MIRROR_MAP[i]"
# Used by data augmentation
MIRROR_MAP = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
              -1, 11, 12, 13, -1, 15, -1, 17, 18, 20,
              19, -1, 22, -1, -1, -1, 26, -1, -1, -1,
              30, -1, -1, 34, 33, 35, 37, 36, 39, 38,
              -1, -1, -1]

# Colors for distribution plot
COLORS = ['tab:blue', 'tab:orange', 'tab:green']

# ---------- Main ---------- #

def main():

    # ---------- Pre-flight checks ---------- #

    # If one of these fail, there is no point continuing

    if not folder_exists(PATH_RAW_FOLDER):
        print("PREFLIGHT ERROR: main(): Raw data folder not found!")
        return

    if not (file_exists(PATH_RAW_TRAIN) or file_exists(PATH_RAW_VALID) or file_exists(PATH_RAW_TEST)):
        print("PREFLIGHT ERROR: main(): Raw data not found!")
        return

    # ---------- Constant Folder Setup ---------- #

    # The program expects these folders to exist

    if not folder_exists(PATH_PREPARED_FOLDER):
        mkdir(PATH_PREPARED_FOLDER)

    if not folder_exists(PATH_MODEL_FOLDER):
        mkdir(PATH_MODEL_FOLDER)

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
        '--data_web',
        type = str,
        nargs = '?',
        default = '',
        help = 'Path to folder containing a set of images.'
    )

    # Show

    parser.add_argument(
        '--show',
        default = None,
        type = str,
        nargs = '*',
        choices = ['images', 'dist', 'pred'],
        help = 'Visualize images, distributions or model predictions.'
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
        '--model_type',
        default = 'VGG16',
        const = 'VGG16',
        type = str,
        nargs = '?',
        choices = ['VGG16', 'LeNet'],
        help = 'Choose a model type/architecture.'
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
        help = 'Model batch size.'
    )

    parser.add_argument(
        '--lrn_rate',
        type = float,
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
    flag_data_web_loaded = False

    # Misc
    flag_force_save = False

    # ---------- Data Setup ---------- #

    path_train = args.data_train
    path_valid = args.data_valid
    path_test = args.data_test
    path_web = args.data_web

    if file_exists(path_train):
        flag_data_train_loaded = True

    if file_exists(path_valid):
        flag_data_valid_loaded = True

    if file_exists(path_test):
        flag_data_test_loaded = True

    # Note: Web images are not pickled
    if not folder_is_empty(path_web):
        flag_data_web_loaded = True

    # ---------- Show Setup ---------- #

    if args.show is not None:

        if len(args.show) > 0:

            flag_show_images = 'images' in args.show
            flag_show_distributions = 'dist' in args.show
            flag_show_predictions = 'pred' in args.show

    if flag_show_distributions:
        if flag_data_train_loaded:
            order_index = 0
        elif flag_data_test_loaded:
            order_index = 1
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

    if flag_model_provided:
        model_path = PATH_MODEL_FOLDER + model_name + '/'
        model_type = args.model_type

        flag_model_evaluate = args.model_evaluate


        # Model hyperparams
        batch_size = args.batch_size
        lrn_rate = args.lrn_rate
        max_epochs = args.max_epochs


        if not folder_is_empty(model_path):
            # The model already exist, no need to train
            flag_model_exists = True
        else:
            # User has created a new model, train it
            flag_model_train = True

        if not folder_exists(model_path):
            mkdir(model_path)

    # ---------- Misc Setup ---------- #

    if args.force_save:
        flag_force_save = True


    # ---------- Metadata Setup ---------- #

    try:
        y_metadata = read_csv(PATH_METADATA)[KEY_METADATA]
    except FileNotFoundError:
        print("Metadata not found!")
        # The program should handle metadata being 'None' without crashing
        y_metadata = None


    # ---------- Argument Checks ---------- #

    # Try to catch argument combinations that either:
    # A) Leads to the program crashing
    # B) Causes the program to do nothing, which can be confusing.
    # The checks listed does NOT form an exhaustive list, and inputs causing A) or B) are possible
    

    # User is trying to show images, but no data is loaded
    if not (flag_data_train_loaded or flag_data_valid_loaded or flag_data_test_loaded or flag_data_web_loaded):
        print("ERROR: No data is loaded, nothing can be done without some data!")
        return

    # User has provided incorrectly named file names for the web data
    # Note: This does not catch incorrect image dimensions, but loading will crash
    if flag_data_web_loaded and (not web_data_file_names_are_valid(path_web)):
        print("ERROR: main(): Images located in ", path_web, 'are incorrectly named!')
        return

    if not flag_model_provided:

        # User has requested evaluation or prediction, but provided no model
        if flag_model_evaluate or flag_show_predictions:
            print("ERROR: main(): You must provide a model in order to do evaluation and/or prediction!")
            return

    if flag_model_provided:

        # User has provided a model and requested training, but no training or validation data is loaded
        if (flag_model_train or flag_force_save) and not (flag_data_train_loaded and flag_data_valid_loaded):
            print("ERROR: main(): You are trying to train your model, but no training and validation data is loaded!")
            return

    if not flag_model_exists:

        # User is trying to evalute a model that does not exist, and no training is requested either
        if (flag_model_evaluate or flag_show_predictions) and not (flag_model_train or flag_force_save):
            print("ERROR: main(): Your are trying to evaluate/predict with a model that does not exist!")
            return

    if flag_model_exists:

        # User is trying to evaluate an existing model, but no testing data is provided.
        if flag_model_evaluate and (not flag_data_test_loaded):
            print("ERROR: main(): You are trying to evalutate your model, but no testing data is loaded!")
            return

        if flag_show_predictions and not (flag_data_test_loaded or flag_data_web_loaded):
            print("ERROR: main(): You are trying to show model predictions, but no relevant data is loaded! ")
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

    if not folder_is_empty(path_web):
        print("Loading web data...")
        X_web, y_web = data_load_web(path_web)
    else:
        X_web, y_web = None, None

    # ---------- Post Load ---------- #

    if flag_data_train_loaded and flag_show_distributions:
        # If the user has loaded a uniform training set (can happen if y_train has undergone augmentation), 
        # use a different set for the order index, otherwise, the plot will appear unstructured and messy
        if dist_is_uniform(y_train):

            if flag_data_test_loaded:
                order_index = 1
            elif flag_data_valid_loaded:
                order_index = 2
            else:
                order_index = 0


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

        # Generate title automatically

        if flag_data_train_loaded:
            title_train = '| Training set (Blue)'
        else:
            title_train = ''

        if flag_data_test_loaded:
            title_test = ' | Testing set (Orange) | '
        else:
            title_test = ''

        if flag_data_valid_loaded:
            title_valid = 'Validation set (Green) |'
        else:
            title_valid = ''

        if not title_test:
            title = title_train + ' | ' + title_valid
        else:
            title = title_train + title_test + title_valid

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

        if model_type == 'VGG16':
            model = VGG16()
        else:
            model = LeNet()

        optimizer = Adam(learning_rate = lrn_rate)

        # flag_force_save: User has the option to force a new training session, even if the model exists 
        if flag_model_train or flag_force_save:
            
            model.compile(optimizer = optimizer, loss = MODEL_LOSS, metrics = MODEL_METRICS)

            early_stopping = EarlyStopping(monitor = MODEL_TRAINING_METRIC, 
                                           patience = MODEL_TRAINING_PATIENCE, min_delta = MODEL_TRAINING_MIN_DELTA, 
                                           mode = MODEL_TRAINING_MODE, restore_best_weights = True)
            history = model.fit(X_train, y_train, batch_size = batch_size, epochs = max_epochs, 
                        validation_data = (X_valid, y_valid), callbacks = [early_stopping])



            print("Saving", model_name, "...")
            model.save_weights(model_path)

            plot_model_history(model_name, history, PATH_IMAGES_README, lrn_rate, batch_size, max_epochs)

            flag_model_is_loaded = True

        if flag_model_evaluate:

            if (not flag_model_is_loaded):

                print("Loading", model_name, "...")
                model.load_weights(model_path).expect_partial()
                model.compile(optimizer = optimizer, loss = MODEL_LOSS, metrics = MODEL_METRICS)

                flag_model_is_loaded = True

            
            model.evaluate(X_test, y_test, batch_size = batch_size)
            print("Done evaluating", model_name, "!")

        if flag_show_predictions:

            if (not flag_model_is_loaded):

                print("Loading", model_name, "...")
                model.load_weights(model_path).expect_partial()
                model.compile(optimizer = optimizer, loss = MODEL_LOSS, metrics = MODEL_METRICS)

                flag_model_is_loaded = True

            if flag_data_test_loaded:
                print("Showing test predictions for", model_name, "...")

                # Load raw data for nicer plots
                X_test_raw, _ = data_load_pickled(PATH_RAW_TEST)
                
                predictions_test = model.predict(X_test)

                n_images = len(X_test)

                indices = np.random.randint(0, n_images, min(n_images, N_PREDICTIONS_MAX))

                if y_metadata is not None:
                    y_pred = predictions_create_titles(predictions_test, y_metadata, indices)
                else:
                    y_pred = predictions_create_titles(predictions_test, y_test, indices)

                X_pred, _, _ = images_pick_subset(X_test_raw, indices = indices, n_images_max = N_PREDICTIONS_MAX)

                show_images(X_pred, titles_bottom = y_pred, title_fig_window = 'Testing set predictions by: ' + model_name, 
                            font_size = 12, n_cols_max = 3, titles_bottom_h_align = 'left', titles_bottom_pos = (34, 7.0))

            if flag_data_web_loaded:
                print("Showing web predictions for", model_name, "...")
                
                X_web_raw, y_web = data_load_web(path_web)

                X_web = pre_process(X_web_raw)

                predictions_test = model.predict(X_web)

                if y_metadata is not None:
                    y_pred = predictions_create_titles(predictions_test, y_metadata, indices = None, n_images_max = N_PREDICTIONS_MAX)
                else:
                    y_pred = predictions_create_titles(predictions_test, y_web, indices = None, n_images_max = N_PREDICTIONS_MAX)

                X_pred = X_web_raw[:min(len(X_web_raw), N_PREDICTIONS_MAX)]

                show_images(X_pred, titles_bottom = y_pred, title_fig_window = 'Web set predictions by: ' + model_name, fig_size = (10, 10), 
                            font_size = 12, n_cols_max = 3, titles_bottom_h_align = 'left', titles_bottom_pos = (34, 7.0))

main()