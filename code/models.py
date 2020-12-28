"""
This file contains any Keras model(s) used in the project
"""

# Imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, Activation, Flatten, Dropout
from tensorflow.keras.layers import Dense

# LeNet

def LeNet():
    """
    Create a LeNet inspired model
    
    Inputs
    ----------
        N/A
        
    Outputs
    -------
    model : tf.keras.Sequential
        LeNet model

    """

    model = Sequential()

    # Conv 32x32x1 -> 28x28x6.
    model.add(Conv2D(filters = 6, kernel_size = (5, 5), strides = (1, 1), padding = 'valid', 
                            data_format = 'channels_last', input_shape = (32, 32, 1)))
    model.add(Activation("relu"))

    # Maxpool 28x28x6 -> 14x14x6
    model.add(MaxPooling2D((2, 2)))

    # Conv 14x14x6 -> 10x10x16
    model.add(Conv2D(filters = 16, kernel_size = (5, 5), padding = 'valid'))
    model.add(Activation("relu"))

    # Maxpool 10x10x16 -> 5x5x16
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Flatten 5x5x16 -> 400
    model.add(Flatten())

    # FC Layer: 400 -> 120
    model.add(Dense(120))
    model.add(Activation("relu"))

    # FC Layer: 120 -> 84
    model.add(Dense(84))
    model.add(Activation("relu"))

    # Dropout
    model.add(Dropout(0.2))
    
    # FC Layer: layer 84-> 43
    model.add(Dense(43))
    model.add(Activation("softmax"))

    return model

# VGG16

def VGG16():
    """
    Create a VGG16 inspired model
    
    Inputs
    ----------
        N/A
        
    Outputs
    -------
    model : tf.keras.Sequential
        Custom model

    """

    model = Sequential()
    norm_axis = -1
    
    # Conv Layer: 32x32x1 -> 32x32x8.
    model.add(Conv2D(filters = 8, kernel_size = (5,5), padding = "same", input_shape = (32, 32, 1)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = norm_axis))

    # Layer: 32x32x1 -> 16x16x8.
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Conv Layer: 16x16x8 -> 16x16x16
    model.add(Conv2D(filters = 16, kernel_size = (3,3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = norm_axis))
    model.add(Conv2D(filters = 16, kernel_size = (3,3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = norm_axis))

    # Layer: 16x16x8 -> 8x8x16
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    # Layer: 8x8x16 -> 8x8x32
    model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = norm_axis))
    model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = norm_axis))
    model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = norm_axis))

    # Layer: 8x8x32 -> 4x4x32
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    # Flatten Layer: 4x4x32 -> 512
    model.add(Flatten())

    # FC Layer: 512 -> 128
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))
    
    # Flatten Layer:: 128 -> 128
    model.add(Flatten())

    # FC Layer: 128 -> 128
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # FC Layer: 128 -> 43
    model.add(Dense(43))
    model.add(Activation("softmax"))
    
    return model