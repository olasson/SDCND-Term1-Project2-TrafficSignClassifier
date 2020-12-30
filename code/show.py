"""
This file contains basic data visualization 
"""

# Custom imports
from code.helpers import data_pick_subset

# General imports
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- Constants ---------- #

# Prevent a user from showing too many images
N_SHOW_IMAGES_MAX = 50 

# Prevvent a user from attempting to show too many distributions at once
N_DISTRIBUTIONS_MAX = 3

# ---------- Functions ---------- #

def show_images(images, titles_top = None, titles_bottom = None, title_fig_window = None, fig_size = (15, 15), font_size = 10, cmap = None, 
                n_cols_max = 5, titles_bottom_h_align = 'center', titles_bottom_v_align = 'top', titles_bottom_pos = (16, 32)):
    """
    Show a set of images
    
    Inputs
    ----------
    images : numpy.ndarray
        A set of images, RGB or grayscale
    titles_top: (None | list)
        A set of image titles to be displayed on top of an image
    titles_bottom: (None | list)
        A set of image titles to be displayed at the bottom of an image
    title_fig_window: (None | string)
        Title for the figure window
    figsize: (int, int)
        Tuple specifying figure width and height in inches
    fontsize: int
        Fontsize of 'titles_top' and 'titles_bottom'
    cmap: (None | string)
        RGB or grayscale
    n_cols_max: int
        Maximum number of columns allowed in figure
    titles_bottom_h_align: string
        Horizontal alignment of 'titles_bottom'
    titles_bottom_v_align: string
        Vertical alignment of 'titles_bottom'
    titles_bottom_pos: (int, int)
        Tuple containing the position of 'titles_bottom'
    titles_bottom_transform: string
        Coordinate system used by matplotlib for 'titles_bottom'


    Outputs
    -------
    plt.figure
        Figure showing 'images' in an (n_rows x n_cols) layout
    
    """

    n_images = len(images)

    if n_images > N_SHOW_IMAGES_MAX:
        print("ERROR: code.show.show_images(): You're trying to show", n_images, "images. Max number of allowed images:", N_SHOW_IMAGES_MAX)
        return

    n_cols = int(min(n_images, n_cols_max))
    n_rows = int(np.ceil(n_images / n_cols))

    fig = plt.figure(title_fig_window, figsize = fig_size)

    for i in range(n_images):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(images[i], cmap = cmap)

        plt.xticks([])
        plt.yticks([])

        if titles_top is not None:
            plt.title(titles_top[i], fontsize = font_size)

        if titles_bottom is not None:
            plt.text(titles_bottom_pos[0], titles_bottom_pos[1], 
                     titles_bottom[i],
                     verticalalignment = titles_bottom_v_align, 
                     horizontalalignment = titles_bottom_h_align,
                     fontsize = font_size - 3)

    plt.tight_layout()
    plt.show()


def show_label_distributions(distributions, labels_metadata = None, order_index = 0, title = None, title_fig_window = None, fig_size = (15, 10), font_size = 6, colors = None):
    """
    Show label distribution
    
    Inputs
    ----------
    distributions: list
        A list of one or up to three label sets -- [labels1, labels2, labels3]
    labels_metadata: (numpy.ndarray | None)
        Labels metadata
    order_index: int
        Determines which distribution will be used to set the class order on the y-axis 
    title: string
        Title for the histogram plot
    title_fig_window: (None | string)
        Title for the figure window
    figsize: (int, int)
        Tuple specifying figure width and height in inches
    fontsize: int
        Integer determining the font size of text in the figure
        
    Outputs
    -------
    plt.figure
        Figure showing label class distribution(s)
    
    """

    if len(distributions) > N_DISTRIBUTIONS_MAX:
        print("ERROR: code.show.show_distributions(): You're trying to show", len(distributions), "distributions. Max number of allowed distributions:", N_DISTRIBUTIONS_MAX)
        return

    if colors is None:
        colors = [None for d in distributions]

    # ---------- Set the desired class order---------- #

    classes, classes_count_order = np.unique(distributions[order_index], return_counts = True)

    # Ensure that the order of classes fits a reverse sort of 'classes_count_order'
    # 'classes_order' determines the order of classes on the y-axis
    classes_order = [tmp for _,tmp in sorted(zip(classes_count_order, classes), reverse  = True)]

    # Perform a reverse sort of 'classes_order'
    classes_count_order = sorted(classes_count_order, reverse = True)

    # ---------- Prepare 'y_ticks' ---------- #

    y_ticks = []
    for class_i in classes_order:
        if labels_metadata is not None:
            y_ticks.append(labels_metadata[class_i])
        else:
            y_ticks.append(class_i)

    plt.figure(title_fig_window, figsize = fig_size)

    for i,distribution in enumerate(distributions):
        if distribution is not None:
            _, classes_count = np.unique(distribution, return_counts = True)

            # Fit the class order of 'distribution' to the desired class order
            classes_count = data_pick_subset(classes_count, classes_order)

            plt.barh(classes, classes_count, color = colors[i])

    plt.yticks(classes, y_ticks, fontsize = font_size)

    plt.xlabel("Number of each class", fontsize = font_size + 14)

    plt.title(title, fontsize = font_size + 20)

    if labels_metadata is None:
        plt.ylabel("Class ID", fontsize = font_size + 14)

    plt.show()

def plot_model_history(model_name, history, path_save = None, lrn_rate = None, batch_size = None, max_epochs = None):
    """
    Plot model history and metadata
    
    Inputs
    ----------
    model_name: string
        Name of the model
    history: Keras History Object
        Model history (output from .fit)
    path_save: (None | string)
        Path to where the plot will be saved. 
    lrn_rate: (None | float)
        Model learning rate
    batch_size: (None | int)
        Model batch size
    max_epochs: (None | int)
        Model max epochs 
        
    Outputs
    -------
    plt.figure
        Figure showing model history and metadata, either shown directly or saved in location 'path_save'
    
    """

    train_log = history.history['loss']
    valid_log = history.history['val_loss']
    
    train_loss = train_log[-1]
    valid_loss = valid_log[-1]
    
    text = "Training/Validation Loss: " + str(round(train_loss, 3)) + '/' + str(round(valid_loss, 3))   
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    c1 = colors[0]
    c2 = colors[1]
    
    fig, ax1 = plt.subplots(figsize = (9, 6))
    
    ax1.set_xlabel('Epochs')    
    ax1.set_ylabel('Loss')

    x = np.arange(1, len(train_log) + 1)
    
    ax1.plot(x, train_log, label = 'Train Loss', color = c1)
    ax1.plot(x, valid_log, label = 'Validation Loss', color = c2)


    stopping_epoch = len(history.history['loss'])

    # ---------- Construct a title for the plot ---------- # 

    model_name_title = 'Model Name: '+ model_name + ' | '

    if lrn_rate is not None:
        lrn_rate_title = 'Lrn rate: ' + str(lrn_rate) + ' | '
    else:
        lrn_rate_title = ''

    if batch_size is not None:
        batch_size_title = 'Batch size: ' + str(batch_size) + ' | '
    else:
        batch_size_title = ''

    if max_epochs is not None:
        epochs_title = 'Stopp/Max (Epoch): ' + str(stopping_epoch) + '/' + str(max_epochs)
    else:
        epochs_title = 'Stopp Epoch: ' + str(stopping_epoch)

    plt.title(model_name_title + lrn_rate_title + batch_size_title + epochs_title)

    # ---------- Misc ---------- #
    
    fig.text(0.5, 0, text,
                verticalalignment = 'top', 
                horizontalalignment = 'center',
                color = 'black', fontsize = 10)
    
    handles, labels = ax1.get_legend_handles_labels()
    
    fig.legend(handles, labels, loc = (0.7, 0.5))
    fig.tight_layout()

    # ---------- Show or save ---------- #
    
    # If the user has opted to save the model history, don't show the plot directly
    if path_save is not None:
        fig.savefig(os.path.join(path_save, model_name), bbox_inches = 'tight')
    else:
        plt.show()
