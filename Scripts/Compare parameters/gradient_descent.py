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
param_default_and_list_dict = {
    "num_units":        [10,    [5, 10, 15, 20]],
    "num_layers":       [1,     [1, 2, 3]],
    "learning_rate":    [0.1,   np.logspace(-3, 1, 5)],
    "s0":               [1,     np.logspace(-1, 3, 5)],
    "alpha":            [0.5,   np.arange(0.5, 1, 0.1)],
    "beta":             [0.5,   np.arange(0.5, 1, 0.1)],
}

# Initialise data set
pass

# Set number of repeats
n_repeats = 3

# Iterate through each experiment (one parameter is varied per experiment)
for var_param_name, (_, var_param_list) in param_default_and_list_dict.items():
    # Initialise dictionary of parameters for this experiment, using defaults
    experiment_params_dict = {
        name: default_val
        for name, (default_val, _)
        in param_default_and_list_dict.items()
    }
    # Initialise results list
    results_list = []

    # Iterate through each value for the parameter under test
    for var_param in var_param_list:
        # Set the value of the parameter under test for this experiment
        experiment_params_dict[var_param_name] = var_param
        # Run experiment, store results
        pass

    # Plot results for experiment with this parameter
    pass
