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
from time import perf_counter
import numpy as np
if __name__ == "__main__":
    import __init__
import plotting


def run_all_experiments(
    all_experiments_dict,
    run_experiment,
    dataset,
    output_dir,
    n_repeats=5,
    verbose=True,
    alpha_plotting=0.5
):
    t_0 = perf_counter()

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
            # Run experiment and store the results
            for i in range(n_repeats):
                np.random.seed(i)
                result = run_experiment(dataset, **this_experiment_dict)
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
            alpha_plotting
        )

    # If verbose, then print total time taken to run all experiments
    if verbose:
        mins, secs = divmod(perf_counter() - t_0, 60)
        if mins > 0:
            print("All tests run in {} mins {:.2f} s".format(int(mins), secs))
        else:
            print("All tests run in {:.2f} s".format(secs))
