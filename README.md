# **Traffic Sign Classifier** 

*by olasson*

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

*This is a revised version of my Traffic Sign Classifier project.*

## Project overview

The majority of the project code is located in the folder `code`:

* [`augment.py`](https://github.com/olasson/SDCND-Term1-Project2-TrafficSignClassifier/blob/master/code/augment.py)
* [`helpers.py`](https://github.com/olasson/SDCND-Term1-Project2-TrafficSignClassifier/blob/master/code/helpers.py)
* [`io.py`](https://github.com/olasson/SDCND-Term1-Project2-TrafficSignClassifier/blob/master/code/io.py)
* [`models.py`](https://github.com/olasson/SDCND-Term1-Project2-TrafficSignClassifier/blob/master/code/models.py)
* [`process.py`](https://github.com/olasson/SDCND-Term1-Project2-TrafficSignClassifier/blob/master/code/process.py)
* [`show.py`](https://github.com/olasson/SDCND-Term1-Project2-TrafficSignClassifier/blob/master/code/show.py)

The main project script is called [`traffic_sign_classifier.py`](https://github.com/olasson/SDCND-Term1-Project1-DetectLaneLines/blob/master/traffic_sign_classifier.py). It contains the implementation of a very simple command line tool.

The results of model training and predictions are found in:
* `images/`

The images shown in this readme are found in 

* `images/`

## Command line arguments

The following command line arguments are defined:
* `--data_train:` Path to a pickled (.p) training dataset.
* `--data_valid:` Path to a pickled (.p) validation dataset.
* `--data_test:` Path to a pickled (.p) testing dataset.
* `--data_web:` Path to a folder containing a set of images.
* `--show:` Visualize one or more datasets. If used, one or more required: `['images', 'dist', 'pred']`. 
* `--prepare:` Prepare one or more datasets for use by a model. Optional (agumentation): `['mirroring', 'rand_tf']`.
* `--model_name:` Name of model.
* `--model_type:` Model type. Optional: `['VGG16', 'LeNet']`. Default: `'VGG16'`.
* `--model_evaluate:` Evaluate model performance on set provided by `--data_test`. Default: `False`.
* `--lrn_rate:` Model optimizer learning rate. Default: `0.001`.
* `--batch_size:` Model batch size. Default: `64`.
* `--force_save:` Override existing prepared data and/or models. Default: `False`.

While all arguments are technically optional, it will not run if no data is provided. The program also checks for "illegal" argument combinations to an extent, and performs some basic "pre-flight" checks.

## Data Exploration

Below is a very basic data overview.

| Dataset   |      # Images      |  # Unique Classes |  Shape |
|----------|:-------------:|------:|------:|
| `train.p` |  34799 | 43 | (32,32,3) |
| `valid.p` |  4410 | 43 | (32,32,3) |
| `test.p` |  12630 | 43 | (32,32,3) |

Next, lets take a look at a random subset of training images. See the `images/` folder for validation and testing images. 

Command: `--data_train './data/train.p' --show 'images'`

<p align="center">
  <img width="80%" height="80%" src="https://github.com/olasson/SDCND-Term1-Project2-TrafficSignClassifier/blob/master/images/rand_images_train.png">
</p>

*Observation 1:* The images have uneven brightness. This should be corrected for in the pre processing step. 

Next, lets compare the class distributions (click on image to enlarge). 

Command: 

`--data_train './data/train.p' --data_valid './data/valid.p' --data_test './data/test.p' --show 'dist'`

<p align="center">
  <img width="80%" height="80%" src="https://github.com/olasson/SDCND-Term1-Project2-TrafficSignClassifier/blob/master/images/raw_dist_comp.png">
</p>

*Observation 2:* Very uneven training distribution, which could lead to overfitting.

## Data Preparation

This section is concerned with preparing the datasets for use by a model.

The command used for creating the "prepared" datasets:

 `--data_train './data/train.p' --data_valid './data/valid.p' --data_test './data/test.p' --prepare 'mirroring' 'rand_tf'`

### Augmentation

This attempts to counter *Observation 2* in the previous section through artificially creating more training images until an uniform distribution is created. 

Relevant code: `code/augment.py`

#### Mirroring

It is possible to mirror certain classes to imitate others. This is "formalized" in the following mirror map found in `traffic_sign_classifier.py`:

    MIRROR_MAP = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                  -1, 11, 12, 13, -1, 15, -1, 17, 18, 20,
                  19, -1, 22, -1, -1, -1, 26, -1, -1, -1,
                  30, -1, -1, 34, 33, 35, 37, 36, 39, 38,
                  -1, -1, -1]

The mirror map defines a mapping where *`Class i` is mirrored to imitate `Class mirror_map[i]`*. For example, "Turn Left Ahead" is mirrored to imitate "Turn Right Ahead". The mirror map is used by `augment_data_by_mirroring()`.

#### Random transformations

By applying one or more (up to four) random transformations, new images can be created from existing ones. The following random transformations are defined in `code/augment.py`:

* `scale_image()`
* `translate_image()`
* `prespective_transform()`
* `rotate_image()`

All transformations preserve the original image dimensions. One or more of these is applied by `random_transforms()` which in turn is called by `augment_data_by_random_transform()`. It will apply random transformation to each class until a target count for each class is reached. 

<p align="center">
  <img width="80%" height="80%" src="https://github.com/olasson/SDCND-Term1-Project2-TrafficSignClassifier/blob/master/images/random_transforms.png">
</p>

For this project, I only applied agumentation to `train.p`. Lets take a look at the distribution after augmentation. 

Command: 

`--data_train './data/prepared_train.p' --data_valid './data/prepared_valid.p' --data_test './data/prepared_test.p' --show 'dist'`

<p align="center">
  <img width="80%" height="80%" src="https://github.com/olasson/SDCND-Term1-Project2-TrafficSignClassifier/blob/master/images/prepared_dist_comp.png">
</p>
            
#### Pre-processing

This step is concerned with establishing a "minimum quality" of data that is fed to the model.

Relevant code: `code/process.py`

#### Histogram Equalization. 

In order to combat uneven brightness, histogram equalization (CLAHE) is applied by `histogram_equalization()`. 

<p align="center">
  <img width="80%" height="80%" src="https://github.com/olasson/SDCND-Term1-Project2-TrafficSignClassifier/blob/master/images/preproc_hist_eq.png">
</p>

The brightness of the images in the bottom row is more equal. Hopefully this will allow the model to focus more on the physical features of the signs, and less on the brightness. 

#### Grayscale

In order to lighten the computation load, grayscale conversion is applied by `grayscale()`. 

#### Normalization 

In order to ensure a set range of values the model has to learn, and in turn (hopefully) cause faster optimizer convergence, normalization is applied by `normalize()`.

## Models

This section defines the models used in the project. 

Relevant code: `code/models.py`

### VGG16 Inspired

The first model, called `VGG16_50_16` is inspired by the [VGG16 architecture](https://neurohive.io/en/popular-networks/vgg16/). A model summary follows below: 

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 32, 32, 8)         208       
    _________________________________________________________________
    activation (Activation)      (None, 32, 32, 8)         0         
    _________________________________________________________________
    batch_normalization (BatchNo (None, 32, 32, 8)         32        
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 16, 16, 8)         0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 16, 16, 16)        1168      
    _________________________________________________________________
    activation_1 (Activation)    (None, 16, 16, 16)        0         
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 16, 16, 16)        64        
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 16, 16, 16)        2320      
    _________________________________________________________________
    activation_2 (Activation)    (None, 16, 16, 16)        0         
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 16, 16, 16)        64        
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 8, 8, 16)          0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 8, 8, 32)          4640      
    _________________________________________________________________
    activation_3 (Activation)    (None, 8, 8, 32)          0         
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 8, 8, 32)          128       
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 8, 8, 32)          9248      
    _________________________________________________________________
    activation_4 (Activation)    (None, 8, 8, 32)          0         
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 8, 8, 32)          128       
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 8, 8, 32)          9248      
    _________________________________________________________________
    activation_5 (Activation)    (None, 8, 8, 32)          0         
    _________________________________________________________________
    batch_normalization_5 (Batch (None, 8, 8, 32)          128       
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 4, 4, 32)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 512)               0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               65664     
    _________________________________________________________________
    activation_6 (Activation)    (None, 128)               0         
    _________________________________________________________________
    batch_normalization_6 (Batch (None, 128)               512       
    _________________________________________________________________
    dropout (Dropout)            (None, 128)               0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 128)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 128)               16512     
    _________________________________________________________________
    activation_7 (Activation)    (None, 128)               0         
    _________________________________________________________________
    batch_normalization_7 (Batch (None, 128)               512       
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 43)                5547      
    _________________________________________________________________
    activation_8 (Activation)    (None, 43)                0         
    =================================================================
    Total params: 116,123
    Trainable params: 115,339
    Non-trainable params: 784


All activation layers are of type `relu`, except for `activation_8` which is `softmax`. 

### LeNet Inspired

The first model, called `LeNet_50_16` is inspired by [LeNet architecture](https://en.wikipedia.org/wiki/LeNet). A model summary follows below:

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 28, 28, 6)         156       
    _________________________________________________________________
    activation (Activation)      (None, 28, 28, 6)         0         
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 6)         0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 10, 10, 16)        2416      
    _________________________________________________________________
    activation_1 (Activation)    (None, 10, 10, 16)        0         
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 5, 5, 16)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 400)               0         
    _________________________________________________________________
    dense (Dense)                (None, 120)               48120     
    _________________________________________________________________
    activation_2 (Activation)    (None, 120)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 84)                10164     
    _________________________________________________________________
    activation_3 (Activation)    (None, 84)                0         
    _________________________________________________________________
    dropout (Dropout)            (None, 84)                0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 43)                3655      
    _________________________________________________________________
    activation_4 (Activation)    (None, 43)                0         
    =================================================================
    Total params: 64,511
    Trainable params: 64,511
    Non-trainable params: 0

All activation layers are of type `relu`, except for `activation_4` which is `softmax`. 

### Training

The user has the option to specify a couple of hyperparameters through the command line namely `lrn_rate`, `batch_size` and `max_epochs`. While the first two are fairly stright forward, `max_epochs` is, as the name implies, not necessarily the number of epochs training will run for. This is due to the Keras callback implemented like so

    MODEL_TRAINING_PATIENCE = 5
    MODEL_TRAINING_MODE = 'max'
    MODEL_TRAINING_METRIC = 'val_accuracy'
    MODEL_TRAINING_MIN_DELTA = 0.001
    
    ...
    
    early_stopping = EarlyStopping(monitor = MODEL_TRAINING_METRIC, 
                                   patience = MODEL_TRAINING_PATIENCE, min_delta = MODEL_TRAINING_MIN_DELTA, 
                                   mode = MODEL_TRAINING_MODE, restore_best_weights = True)

 
In plain english, this callback does the following: *Stop the traning if there has not been a `min_delta` improvement in the metric `monitor` for `patience` epochs.* With the specified values: *Stop the traning if there has not been a `0.001` improvement in the metric `val_accuracy` for `5` epochs.*

This will prevent the training from running while no "meaningful" improvement in accuracy is achieved. It is worth noting that this in no way guarantees the best training. 

## Results

A simple naming standard was devised to differentiate the models: `<model_type>_<lrn_rate_decimals>_<batch_size>`. As an example, `VGG16_001_64` reads as *A VGG16 model trained with (lrn_rate = 0.001)  and (batch_size = 64)*. I left `max_epochs = 50` for the entire project and for all models. 

In terms of the command line tool there is one important difference. Since `VGG16` is the default, it is necessary to specify the model type for LeNet. 

* `(...) --model_name  'VGG16_001_64' ` (Sufficient)
* `(...) --model_name  'LeNet_001_64' --model_type 'LeNet' ` (Sufficient)

Below is the results for all models trained. 

| Model Name    | Stop Epoch/Max Epoch  | Training Loss | Validation Loss  | Testing Set accuracy |  
| :---        |    :----:   |          ---: |          ---: |         ---: |
| `VGG16_0001_128`   |    32/50  |  0.2020 |  0.1280 |  0.9430 |
| `VGG16_001_64`   |    16/50  |  0.0630 |  0.057 |  0.9710 |
| `VGG16_01_32`   |    14/50  |  0.1340 |  0.0870 |  0.9629 |
| `LeNet_0001_128`   |    45/50  |  0.2300 |  0.1520 |  0.9284 |
| `LeNet_001_64`   |    20/50  |  0.0490 |  0.177 |  0.9549 |
| `LeNet_01_32`   |    6/50  |  1.1210 |  0.8620 | 0.7914  |

Plots for all models can be found in the `images/` folder. 

Overall, the model with the best performance seems to be `VGG16_001_64`.

The command line tool support visualizing a subset of the test set predictions. It will show max 15 predictions, with the top 5 probabilities for each image. 

Command:  `--data_test'./data/prepared_test.p/' --model_name  'VGG16_001_64' --show 'pred' `

<p align="center">
  <img width="80%" height="80%" src="https://github.com/olasson/SDCND-Term1-Project2-TrafficSignClassifier/blob/master/images/VGG16_001_64_pred_test.png">
</p>


Some random images was downloaded from the web, prepared and fed to the model. They werre cropped to 32x32. 

Command: `--data_web './data/images/' --model_name  'VGG16_001_64' --show 'pred' `

<p align="center">
  <img width="80%" height="80%" src="https://github.com/olasson/SDCND-Term1-Project2-TrafficSignClassifier/blob/master/images/VGG16_001_64_pred_web.png">
</p>

The model works quite well on the web images. The high confidence seems to indicate that the model has generalized well to new images. 



