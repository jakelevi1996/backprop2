
# backprop2

Implementation of efficient 2nd order methods for training neural networks (work in progress)

## File structure description

This repository contains the following modules:

- `models.py`: contains the `NeuralNetwork` class, which itself contains methods for initialisation, evaluation, and gradient calculations
- `layer.py`: contains the `NeuralLayer` class, which is used by the `NeuralNetwork` class to represent each layer in the network, and the operations that it needs to perform
- `activations.py`: contains activation functions and their derivatives, and an ID for each activation function (allowing them to be saved along with neural network models)
- `errors.py`: contains error functions and their derivatives
- `data.py`: contains a DataSet interface class for saving and loading data sets, and subclasses for generating different types of synthetic data
- `optimisers.py`: contains optimisation routines for optimising models for a particular dataset
- `plotting.py`: contains functions to plot data-sets, model predictions, and training curves
- `training.py`: top-level module for loading datasets, training models, and plotting results


## TODO

2020-07-19:

- Add full unit testing coverage (and remove `if __name__ == "__main__"` blocks)
- Store number of weights and biases in layer attribute, instead of calculating from input dimension and output dimension
- Implement back-propagation of 2nd order gradients
- Implement block-generalised-Newton optimisation
- General tidying up of code and comments
- Implement DBS + convergence metric

2020-05-01 and before:

- Add datasets for Gaussians and sum-of-Gaussians, binary circles, binary multiple circles
- Get binary cross entropy working for classification data-sets, train classification, and add plotting function for 2D binary classification
- Add multi-class classification error function, data-sets, training, and plotting
- (Low priority) investigate how number of layers and units per layer affects success/failure for learning different data sets, EG AND, XOR, circle, multiple circles, etc
- Add PSO optimisation routine and compare to SGD with 2-way tracking
  - Add PSO with 2-way tracking for each particle?
- Add timeout option to optimisation routines, and replace `eval_every` with `eval_every_n_s` based on time instead of iteration (including `while next_eval < perf_counter(): next_eval += eval_every_n_s` to make sure evaluations don't play catch-up)
- Add plotting function for step sizes
- Add plotting function which creates a movie (or gif) showing model predictions evolving over time as the network trains, and compare for different optimisation methods (this may recycle pre-existing plotting functions, and just combine plots into a movie)
- Commit some results graphs to git and display on the readme by including links to GitHub URLs
- Remove all commented code
- Make sure all docstring are sufficiently detailed for the function
- Update readme
- Add efficient second order training methods
- Add `DataSet` subclasses for classification
- Add functionality to `training` module to perform systematic comparison between learning curves for different hyperparameters and activation functions (use multiprocessing?)
- Add saving and loading to `NeuralNetwork` class
- Add `Results` class to training module, which saves and restores learning curves, learned parameters, hyperparameters, etc.
- Investigate standard deviation of gradients throughout training; also natural means

## Usage example

*Coming soon*.
