
# backprop2

Implementation of efficient 2nd order methods for training neural networks (work in progress)

## Performance of PBGN

TODO

## Discussion of performance

TODO

## File structure description

This repository contains the following modules (TODO: this section is outdated, and needs to be updated):

- `models.py`: contains the `NeuralNetwork` class, which itself contains methods for initialisation, evaluation, and gradient calculations
- `layer.py`: contains the `NeuralLayer` class, which is used by the `NeuralNetwork` class to represent each layer in the network, and the operations that it needs to perform
- `activations.py`: contains activation functions and their derivatives, and an ID for each activation function (allowing them to be saved along with neural network models)
- `errors.py`: contains error functions and their derivatives
- `data.py`: contains a DataSet interface class for saving and loading data sets, and subclasses for generating different types of synthetic data
- `optimisers.py`: contains optimisation routines for optimising models for a particular dataset
- `plotting.py`: contains functions to plot data-sets, model predictions, and training curves
- `training.py`: top-level module for loading datasets, training models, and plotting results


## TODO

### 2020-10-15

- Implement initialisers
- Implement unit tests for initialisers
- In a new branch, refactor minimise into a wrapper for a method of a class, which passes itself to the Result.update method, and update Result to be configurable with new columns (this will need good unit tests)
  - **Better idea**: Just let the `Result.update` method accept a `**kwargs` argument, and store a list of `Column` objects (or dictionaries) in `Result`, each of which prints a value/returns a string depending on its width, format specifier, and using a `lambda` expression to extract a value from the `**kwargs` inputs to `Result.update`
    - Also add a method `_add_default_columns` to `Result`, which adds the default columns, and a default argument `add_default_columns=True` to the `Result` initialiser which adds the default columns and specifiers (time, iteration, train error, test error, etc)
    - Remove `s = 1 if line_search is None else line_search.s` from `minimise`, and just pass the `LineSearch` object object as part of `**kwargs` to `Result.update`
    - Add an `add_column(column: Column)` method to `Result` to add an arbitrary column, but also add convenience methods to add specific columns, EG `add_line_search_column` and `add_dbs_column`, with the current width and format specifiers as default arguments which are passed to the initialiser of `Column`
    - The `Column` class should have methods `get_title_str` and `get_value_str` (the latter will accept the `**kwargs` from `Result.update`)?
- Implement dynamic batch sizing
- Update all scripts to reflect new interfaces and verify
- (In a new branch) Update 2D-ND sinusoidal data inputs to be uniformly distributed between specified limits in the XY-plane; when plotting, do a scatter plot, with the Z-value being represented by the marker colour. Plot predictions the same as before (`pcolormesh` on a uniform grid)
- Update `NeuralNetwork.__call__` method according to its docstring. Also, if `w is not None and x is None and t is None`, then `return self`

### 2020-10-11

- I wonder if the use of `einsum` could be having a negative effect o the performance of PBGN vs GD, since GD seemed to perform worse when using `einsum` instead of `matmul` for transposed matrix multiplication? Might be worth branching and trying to reimplement Hessian calculations without `einsum`, and seeing if this improves the performance of PBGN
- Add performance comparison of optimisers to README, and comparison with previous repo, and discussion
- Investigate how batch-size affects final performance for large 2D sinusoidal data set
- Implement DBS
- Improvements to `run_all_experiments`:
  - In plotting function called by `run_all_experiments`, plot mean (and error bars for standard deviation?) for each value of the parameter under test
  - In `run_all_experiments`, instead of storing the results in 2 lists, store the results in a dictionary, where the keys are the values of the parameter under test, and the values are a list of results for each repeat of that value of the parameter under test
  - Write function for automatically calling `run_all_experiments` in a loop, each time changing the default value of each parameter to the value which has the lowest mean error (or lowest given linear combination of mean and standard deviation) until the default parameter values converge to a local optimum, or a maximum number of repeats is exceeded
    - Use these locally optimum parameter values to determine the best values for PBGN and for GD with line-search, and then compare learning curvves for PBGN vs GD with these locally optimum parameters
  - Fix weird x-axis labels

### 2020-10-04

- Why is PBGN doing bad without line-search?
  - How frequently is max-step being exceeded?
    - Could the Results class be configured to collect data for this only when specified in the constructor, which could then be plotted?
      - The minimise function could be refactored into a class (or a wrapper for a method of a class), and the instance could pass itself to the `Result.update` method. Alternative results to be collected could be specified through `lambda` expressions which accept a `Minimiser` instance as an argument
  - What about if max-step is increased/learning rate is decreased?
- Implement scripts to compare parameters for gradient descent and for PBGN
- Compare performance for different batch-sizes

### 2020-09-27

- Implement Taylor-Newton method
- Implement SBGN (serial instead of parallel)
- Include section in this README about what questions are answered by this repository
- Allow optimisers to be passed to an `optimise` method of the `NeuralNetwork` class, possibly refactoring all optimisers into a class
- Refactor class in `activations` and `errors` modules to be private, and expose public instantiations of those classes
- Update Scripts/compare_gd_pbgn.py to use refactored classes

### 2020-09-20

- Finish script to make plots of different parameter combinations
- Plot training curves for PBGN and SGD on 2D sinusoidal data with batches
- Plot 2D sinusoidal predictions for different optimisers

### 2020-09-13

- Add docstrings to all modules, functions and methods
- Plot training curves of GD vs PBGN, LS vs no LS, and compare
- Plot predictions for PBGN 1D and 2D sinusoids
- Plot script to compare parameters for PBGN, and another equivalent script for GD
- Add unit tests for Terminator and Evaluator classes
- ***Why is PBGN doing badly without line-search? **Is it possible to test when max_step is being exceeded?**

### 2020-09-06

- Add plotting function to visualise activations in each layer for a given input
- Add plotting function to visualise distribution of gradient values in each layer
- Make minimisation functions re-entrant (EG need to allow Result object to be an optional argument, and not re-initialise it if it is given)?

### 2020-08-31

- Finish separating functions from the `training` module into scripts, and remove `if __name__ == "__main__"` block from `training` module
- Add generic training function to the `training` module, including a CLI to configure training runs
- Add function to `plotting` module to plot an arbitrary `Result` attribute against iteration number (for example step size, number of line-search steps per iteration, etc), and compare against different experiments
- Make `Result` class configurable with different columns and column widths
- Add script to plot results (learning curves and predictions) for 2D-to-ND sinusoidal regression
- Support different batch-sizes (but also maintain support for not using the full-training set for each batch; could maybe implement a new BatchSizer parent class and subclasses in a batchsizer module to make this configurable)
- Finish script to compare different parameters for gradient descent with line-search
- Implement saving and loading of NeuralNetwork class
- DBS

### 2020-07-19

- Add full unit testing coverage (and remove `if __name__ == "__main__"` blocks)
- Allocate numpy arrays in `NeuralNetwork` and `NeuralLayer` class during initialisation, and use numpy `out` argument, to improve performance... actually, is this a sensible thing to do for all arrays? For ones which depend on the number of data points, this will vary depending on the batch size
- General tidying up of code and comments
- Implement DBS + convergence metric

### 2020-05-01 and before

- Add datasets for Gaussians and sum-of-Gaussians, binary circles, binary multiple circles
- Get binary cross entropy working for classification data-sets, train classification, and add plotting function for 2D binary classification
- Add multi-class classification error function, data-sets, training, and plotting
- (Low priority) investigate how number of layers and units per layer affects success/failure for learning different data sets, EG AND, XOR, circle, multiple circles, etc
- Add **PSO** optimisation routine and compare to SGD with 2-way tracking
  - Add PSO with 2-way tracking for each particle?
- Add **adam** optimisation
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
