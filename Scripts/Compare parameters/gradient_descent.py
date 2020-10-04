"""
Find best parameters for gradient descent with line-search, by trying different
combinations and plotting the results. The different parameters that are
compared are:
-   number of units
-   number of layers
-   learning_rate
-   s0 (initial step size)
-   alpha (threshold for backtracking)
-   beta (ratio of changes in step size)

Results are compared by plotting final performance after a fixed length of time
allowed for optimisation.

TODO:
-   Need to add Terminator classes, so optimisation can run until time runs out
-   Add argparse wrapper for this class, so experiments can be configured from
    the command line
"""
import numpy as np

# Initialise dictionary of parameter names, default values, and values to test
# TODO: activation functions (need to allow non-numerical parameters, which plot
# as bar plots)
all_experiments_dict = {
    "num_units":            {"default": 10,    "range": [5, 10, 15, 20]},
    "num_layers":           {"default": 1,     "range": [1, 2, 3]},
    "log10_learning_rate":  {"default": -1,    "range": np.linspace(-3, 1, 5)},
    "log10_s0":             {"default": 0,     "range": np.linspace(-1, 3, 5)},
    "alpha":                {"default": 0.5,   "range": np.arange(0.5, 1, 0.1)},
    "beta":                 {"default": 0.5,   "range": np.arange(0.5, 1, 0.1)},
}

# Initialise data set
pass

# Set number of repeats
n_repeats = 3


# Iterate through each experiment (one parameter is varied per experiment)
for var_param_name, var_param_dict in all_experiments_dict.items():
    # Initialise dictionary of parameters for this experiment, using defaults
    this_experiment_dict = {
        param_name: param_dict["default"]
        for param_name, param_dict
        in all_experiments_dict.items()
    }
    # Initialise results list
    results_param_val_list = []
    results_min_error_list = []

    # Iterate through each value for the parameter under test
    for var_param_value in var_param_dict["range"]:
        # Set the value of the parameter under test for this experiment
        this_experiment_dict[var_param_name] = var_param_value
        # Run experiment, store results
        for i in range(n_repeats):
            np.random.seed(i)
            result = run_experiment(sin_data, **this_experiment_dict)
            results_param_val_list.append(var_param_value)
            results_min_error_list.append(min(result.test_errors))

    # Plot results for experiment with this parameter
    pass
