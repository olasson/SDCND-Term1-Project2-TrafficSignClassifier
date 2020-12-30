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
* `images

The images shown in this readme are found in 

* `images`

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

While all arguments are techincally optional, it will not run if no data is provided. The program also checks for "illegal" argument combinations to an extent. 
