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

*Observation* The images have uneven brightness. This should be corrected for in the pre processing step. 

Next, lets compare the class distributions

<p align="center">
  <img width="80%" height="80%" src="https://github.com/olasson/SDCND-Term1-Project2-TrafficSignClassifier/blob/master/images/raw_dist_comp.png">
</p>

*Observation 2:* Certain image classes can be mirrored to imitate other classes. For example, "Turn Left Ahead" can be mirroried to imitate "Turn Right Ahead".

## Data Preparation

### Augmentation

### Pre-processing

## Models

### VGG16

### LeNet

## Results

## Project Shortcomings

