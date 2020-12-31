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

### VGG16

### LeNet

### Training

## Results

## Project Shortcomings

