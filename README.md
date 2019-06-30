# Introduction
This repo is my toy-project to learn how to use TensorFlow.

# Installation
This project was tested with the following setup:
* Ubuntu 16.04
* Python 3.5.2
* TensorFlow 1.13.1

All the following instructions assume you have already checked out this repo and you are running from the base folder.

## Python environment
We make use of [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) to set up our Python environment. Refer to the link to install it and then create a new environment (we call it `mnist`) by executing:
```commandline
mkvirtualenv -p python3 mnist
```
Install all the required packages by pip-installing them:
```commandline
pip install -r requirements.txt
```

## Environment variables
This framework requires you to define some environment variables to ease access to the file system:
* `MNIST_DATA_DIR`: path to the folder where the data (images and metadata files) will be saved;
* `MNIST_BASE_LOG_DIR`: path to the folder where the trained models and any other file generated during an experiment will be saved;
* `MNIST_DATASET_DEF_DIR`: path to the folder where the JSON files defining the datasets will be saved.

To define these variables append the following lines to your `.bashrc` file:
```commandline
export MNIST_DATA_DIR="/path/to/data_dir"
export MNIST_BASE_LOG_DIR="/path/to/base_log_dir"
export MNIST_DATASET_DEF_DIR="/path/to/dataset_def_dir"
```

# Setup
In this section we explain how to set up all the necessary to run this framework, like downloading the data and creating the datasets.

We download all the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset (70000 images) and store it in a standardized format.
Each sample consists of a pair (image, metadata): the image is stored in PNG format and the metadata (e.g. its class label) is stored in a JSON file. Samples are assigned unique string IDs to identify them.

In order to avoid filling up a single folder with 140000 files (70000 images and 70000 metadata files), we create a system of subfolders based on hashed IDs; in this way each subfolder level will contain at most 256 elements.

The `run_dataset_setup.py` script performs the following actions:
1. downloads the MNIST data;
1. stores the data according to the a standard format;
1. creates the dataset definition files containing the training, validation and testing samples.

Run:
```commandline
python -m apps.run_dataset_setup
```

If you want to edit the default configurations (for example the train-val-test splits) edit the attributes of the classes `mnist.config.GeneralConfig` and `mnist.config.SetupConfig`.

# Usage
## Training
1. Edit the configuration module `mnist.config` as a first thing. In particular, make sure to set the experiment code in `mnist.config.ExperimentConfig.EXPERIMENT_CODE`, as it will be used to identify this model also at evaluation and inference time.
1. Define the input pipeline (such as data augmentation) in the `mnist.data.preprocessing` package.
1. Tweak your model architecture in the `mnist.ml.model` package.
1. Run the experiment by:
```commandline
python -m apps.run_experiment
```

## Evaluation
1. Edit the configuration module `mnist.config`. Use the `mnist.config.ExperimentConfig.EXPERIMENT_CODE` field to specify which model to evaluate.
1. Run the evaluation by:
```commandline
python -m apps.run_evaluation
```

## Inference
1. Edit the configuration module `mnist.config`. Use the `mnist.config.ExperimentConfig.EXPERIMENT_CODE` field to specify which model to use for inference.
1. Run the evaluation by:
```commandline
python -m apps.run_inference --show /path/to/image1.png /path/to/image2.png /path/to/image3.png
```
