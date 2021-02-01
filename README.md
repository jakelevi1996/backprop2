
# backprop2

Implementation of efficient 2nd order methods for training neural networks (work in progress)

## Usage example of training with 2nd order methods

```
>>> import numpy as np
>>> import data, models, optimisers, plotting
>>>
>>> np.random.seed(1926)
>>> sin_data = data.Sinusoidal()
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

## TODO

The to-do list for this project is in the form of JIRA issues which can be found [here](https://jakelevi1996.atlassian.net/browse/MR) (for those who have permission).
