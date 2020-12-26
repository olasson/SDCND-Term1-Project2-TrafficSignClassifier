
from code.helpers import data_pick_subset

import matplotlib.pyplot as plt
import numpy as np

# Prevent a user from showing too many images
N_SHOW_IMAGES_MAX = 50 

# Prevvent a user from attempting to show too many distributions at once
N_DISTRIBUTIONS_MAX = 3
COLORS = ['blue', 'green', 'red']

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


def show_label_distributions(distributions, labels_metadata = None, order_index = 0, title = None, title_fig_window = None, fig_size = (15, 10), font_size = 6):
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

    # ---------- Set the desired class order---------- #

    classes, classes_count_order = np.unique(distributions[order_index], return_counts = True)

    # Ensure that the order of classes fits a reverse sort of 'classes_count_order'
    # 'classes_1' determines the order of classes on the y-axis
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

            plt.barh(classes, classes_count, color = COLORS[i])

    plt.yticks(classes, y_ticks, fontsize = font_size)

    plt.xlabel("Number of each class", fontsize = font_size + 14)

    plt.title(title, fontsize = font_size + 20)

    if labels_metadata is None:
        plt.ylabel("Class ID", fontsize = font_size + 14)

    plt.show()