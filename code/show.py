import matplotlib.pyplot as plt
import numpy as np

MAX_NUMBER_OF_IMAGES = 50

def show_images(images, titles_top = None, titles_bottom = None, fig_size = (15, 15), font_size = 10, cmap = None, title_window = '0',
                n_cols_max = 5, titles_bottom_h_align = 'center', titles_bottom_v_align = 'top', titles_bottom_pos = (16, 34)):
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

    if n_images > MAX_NUMBER_OF_IMAGES:
        print("ERROR: code.show.show_images(): You're trying to show", n_images, "images. Max number of allowed images is", MAX_NUMBER_OF_IMAGES)
        return

    n_cols = int(min(n_images, n_cols_max))
    n_rows = int(np.ceil(n_images / n_cols))

    plt.figure(title_window, figsize = fig_size)


    for i in range(n_images):
        ax = plt.subplot(n_rows, n_cols, i + 1)
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