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
import os
from time import perf_counter
import numpy as np
if __name__ == "__main__":
    import __init__
from models import NeuralNetwork
import activations, data, optimisers, plotting

t_0 = perf_counter()

# Get name of output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs", "Gradient descent")
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

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
    "act_func":             {
        "default": activations.Gaussian(),
        "range": [
            activations.Gaussian(),
            activations.Cauchy(),
            activations.Logistic(),
            activations.Relu(),
        ]
    },
}

# Initialise data set
sin_data = data.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)

# Set number of repeats and alpha
n_repeats = 5
alpha = 0.5
verbose = True

# Define function to be run as the experiment
def run_experiment(
    dataset,
    num_units,
    num_layers,
    log10_learning_rate,
    log10_s0,
    alpha,
    beta,
    act_func
):
    n = NeuralNetwork(
        input_dim=1,
        output_dim=1,
        num_hidden_units=[num_units for _ in range(num_layers)],
        act_funcs=[act_func, activations.Identity()]
    )
    result = optimisers.gradient_descent(
        n,
        dataset,
        learning_rate=pow(10, log10_learning_rate),
        terminator=optimisers.Terminator(t_lim=3),
        evaluator=optimisers.Evaluator(t_interval=0.1),
        line_search=optimisers.LineSearch(
            s0=pow(10, log10_s0), 
            alpha=alpha, 
            beta=beta
        )
    )
    return result


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
    if verbose:
        h_line = "*" * 50
        msg = "Plotting result for {}".format(var_param_name)
        print("", h_line, msg, h_line, "", sep="\n")
    plotting.simple_plot(
        results_param_val_list,
        results_min_error_list,
        var_param_name,
        "Minimum test error",
        "Varying parameter {}".format(var_param_name),
        output_dir,
        alpha
    )

if verbose:
    mins, secs = divmod(perf_counter() - t_0, 60)
    if mins > 0:
        print("All tests run in {} mins {:.2f} s".format(int(mins), secs))
    else:
        print("All tests run in {:.2f} s".format(secs))
