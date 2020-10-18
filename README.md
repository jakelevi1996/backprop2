
# backprop2

Implementation of efficient 2nd order methods for training neural networks (work in progress)

## Usage example of training with 2nd order methods

```
>>> import numpy as np
>>> import data, models, optimisers, plotting
>>>
>>> np.random.seed(1926)
>>> sin_data = data.SinusoidalDataSet1D1D()
>>> model = models.NeuralNetwork(input_dim=1, output_dim=1, num_hidden_units=[10])
>>> result = optimisers.Result()
>>> ls = optimisers.LineSearch()
>>> result.add_column(optimisers.results.columns.StepSize(ls))
>>> optimisers.generalised_newton(model, sin_data, result=result, line_search=ls)

Performing test "Unnamed experiment"...
Iteration | Time (s) | Train error | Test error  | Step Size
--------- | -------- | ----------- | ----------- | ----------
        0 |    0.000 |     0.62186 |     0.63667 |     1.0000
      100 |    0.119 |     0.22288 |     0.24831 |     0.5000
      200 |    0.228 |     0.19616 |     0.23234 |     0.2500
      300 |    0.335 |     0.16916 |     0.22942 |     0.1250
      400 |    0.444 |     0.12387 |     0.19325 |     0.5000
      500 |    0.557 |     0.10104 |     0.16789 |     0.5000
      600 |    0.665 |     0.08514 |     0.15279 |     0.2500
      700 |    0.777 |     0.08031 |     0.14806 |     0.5000
      800 |    0.891 |     0.07729 |     0.14504 |     0.5000
      900 |    0.997 |     0.07523 |     0.14342 |     0.2500
     1000 |    1.102 |     0.07356 |     0.14281 |     0.5000
--------------------------------------------------
Test name                      = Unnamed experiment
Total time                     = 1.1029 s
Total iterations               = 1,000
Average time per iteration     = 1.1029 ms
Average iterations per second  = 906.7

Result('Unnamed experiment')
>>> plotting.plot_training_curves([result])
```

Output image:

![PBGN earning curve](https://raw.githubusercontent.com/jakelevi1996/backprop2/master/Results/Learning%20curves/Learning%20curves.png "PBGN learning curve")

## Performance of PBGN

TODO

## Discussion of performance

TODO

## Unit tests

To run unit tests, use the command `pytest ./Tests --durations 5`:

```
$ pytest ./Tests --durations 5
========================================================================== test session starts ========================================================================== 
platform win32 -- Python 3.7.6, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: C:\Users\Jake\Documents\Programming\backprop2
collected 236 items                                                                                                                                                       

Tests\test_activations.py ...................................................                                                                                      [ 21%] 
Tests\test_batch_sizes.py .......................................                                                                                                  [ 38%] 
Tests\test_data.py ........................                                                                                                                        [ 48%] 
Tests\test_errors.py ............                                                                                                                                  [ 53%] 
Tests\test_line_search.py ...                                                                                                                                      [ 54%] 
Tests\test_network.py ..............................................                                                                                               [ 74%] 
Tests\test_network_errors.py ........................                                                                                                              [ 84%] 
Tests\test_optimiser.py .........                                                                                                                                  [ 88%] 
Tests\test_plotting.py ......                                                                                                                                      [ 90%] 
Tests\test_result.py ......................                                                                                                                        [100%] 

======================================================================= slowest 5 test durations ======================================================================== 
0.48s call     Tests/test_plotting.py::test_plot_2D_nD_regression[1743-3]
0.29s call     Tests/test_activations.py::test_plotting[act_func0]
0.28s call     Tests/test_plotting.py::test_plot_2D_nD_regression[1814-1]
0.28s call     Tests/test_plotting.py::test_plot_training_curves
0.15s call     Tests/test_errors.py::test_plotting[error_func0]
========================================================================== 236 passed in 4.68s ==========================================================================
```

## TODO (yes, these to-do lists should be on Jira, and no, I don't intend to copy them over in the near future)

### 2020-10-15

- Implement initialisers
- Implement unit tests for initialisers
- Store an attribute in `NeuralNetwork` for the DBS metric which can be processed as a column when `model` is passed to `Result.update`, as well as `optimisers.batch.DynamicBatchSize`
- Implement dynamic batch sizing
- Update all scripts to reflect new interfaces and verify
- (In a new branch) Update 2D-ND sinusoidal data inputs to be uniformly distributed between specified limits in the XY-plane; when plotting, do a scatter plot, with the Z-value being represented by the marker colour. Plot predictions the same as before (`pcolormesh` on a uniform grid)
  - Replace multiple SinusoidalDataset classes with a single class
- Update `NeuralNetwork.__call__` method according to its docstring. Also, if `w is not None and x is None and t is None`, then `return self`
- Implement saving and loading Result objects
- Update plotting functions for new Sinusoidal dataset objects
- Add plotting function to plot Result Column values over time

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
