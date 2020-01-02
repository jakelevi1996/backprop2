
# backprop2

Implementation of efficient 2nd order methods for training neural networks (work in progress)

## File structure description

This repository contains the following modules:

- `models.py`: contains the neural network class, which itself contains methods for initialisation, evaluation, and gradient calculations
- `activations.py`: contains activation functions and their derivatives, and an ID for each activation function (allowing them to be saved along with neural network models)
- `errors.py`: contains error functions and their derivatives
- `data.py`: contains a DataSet interface class for saving and loading data sets, and subclasses for generating different types of synthetic data
- `optimisers.py`: contains optimisation routines for optimising models for a particular dataset
- `plotting.py`: contains functions to plot data-sets, model predictions, and training curves
- `training.py`: top-level module for loading datasets, training models, and plotting results

## TODO

- Update readme
- Add efficient second order training methods
- Add backtracking line search to first-order methods
- Update `training` and `model` modules to plot learning curves (test data and/or training data)
- Add `DataSet` subclasses for classification
- Add functionality to `training` module to perform systematic comparison between learning curves for different hyperparameters and activation functions (use multiprocessing?)
- Add saving and loading to `NeuralNetwork` class
- Add `Results` class to training module, which saves and restores learning curves, learned parameters, hyperparameters, etc.
- Investigate standard deviation of gradients throughout training; also natural means

## Usage example

*Coming soon*.
