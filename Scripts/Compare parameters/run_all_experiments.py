"""
Module containing the test-neutral run_all_experiments function for comparing
parameters. This function should be called by wrapper scripts for particular
optimisers and sets of parameters to test.
"""
import os
import sys
from datetime import datetime
from traceback import print_exception
from time import perf_counter
import numpy as np
if __name__ == "__main__":
    import __init__
import optimisers, plotting

def print_error_details(experiment_dict):
    """
    Print details of an error both to STDOUT and to a time-stamped error log
    file
    """
    t = datetime.now()
    error_filename = "{} Error log.txt".format(t).replace(":", ".")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    error_dir = os.path.join(current_dir, "Error logs")
    if not os.path.isdir(error_dir):
        os.makedirs(error_dir)
    error_file_path = os.path.join(error_dir, error_filename)
    with open(error_file_path, "w") as error_file:
        for f in [error_file, None]:
            print("Parameter details:", file=f)
            for key, value in experiment_dict.items():
                print("{}: {}".format(repr(key), value), file=f)
            print("*" * 50, file=f)
            print("Error details:", file=f)
            print_exception(*sys.exc_info(), file=f)

def run_all_experiments(
    all_experiments_dict,
    run_experiment,
    dataset,
    output_dir,
    n_repeats=5,
    verbose=True,
    alpha_plotting=0.5
):
    """
    Run experiments for each parameter specified in all_experiments_dict, and
    save plots for each parameter under test to disk.

    Inputs:
    -   all_experiments_dict: dictionary in which each key is the name of a
        parameter that will be passed to run_experiment, and each value is a
        dictionary containing a "default" key and a "range" key, for the default
        value to use when testing other parameters, and the list of values to
        try when testing this parameter
    -   run_experiment: a callable that should accept a Dataset object as the
        first positional argument and all of the keys in all_experiments_dict as
        keyword arguments, run the corresponding experiment, and return a Result
        object
    -   dataset: a Dataset object to pass to run_experiment
    -   output_dir: directory in which to save plots for each parameter under
        test
    -   n_repeats: number of repeats for each parameter combination
    -   verbose: whether to print out each time a plot is being made, and total
        time taken
    -   alpha_plotting: transparency value for markers in the output plots

    Outputs:
    -   None
    -   (plots for each parameter under test are saved to disk)
    """
    t_0 = perf_counter()

    # Iterate through each experiment (one parameter is varied per experiment)
    for var_param_name, var_param_dict in all_experiments_dict.items():
        # Initialise dictionary for this experiment using default parameters
        this_experiment_dict = {
            param_name: param_dict["default"]
            for param_name, param_dict
            in all_experiments_dict.items()
        }
        # Initialise results lists
        results_param_val_list = []
        results_final_error_list = []

        # Iterate through each value of the parameter under test
        for var_param_value in var_param_dict["range"]:
            # Set the value of the parameter under test for this experiment
            this_experiment_dict[var_param_name] = var_param_value
            # Run experiment and store the results
            for i in range(n_repeats):
                np.random.seed(i)
                try:
                    result = run_experiment(dataset, **this_experiment_dict)
                    results_param_val_list.append(var_param_value)
                    test_errors = result.get_values(
                        optimisers.results.columns.TestError
                    )
                    results_final_error_list.append(test_errors[-1])
                except:
                    print_error_details(this_experiment_dict)

        # Plot results for experiment with this parameter
        if verbose:
            h_line = "*" * 50
            msg = "Plotting result for {}".format(var_param_name)
            print("", h_line, msg, h_line, "", sep="\n")
        # Test if all x-values are ints or floats (NOTE: elements of numpy
        # arrays are SUBCLASSES of float, but comparing the type directly to
        # float or int will return False!!!)
        all_numeric = all(
            type(x) in [int, float, np.float64]
            for x in results_param_val_list
        )
        # If not all x-values are ints or floats, then format as strings
        if not all_numeric:
            results_param_val_list = [
                repr(x).replace("activation function", "").rstrip()
                for x in results_param_val_list
            ]
        plotting.simple_plot(
            results_param_val_list,
            results_final_error_list,
            var_param_name,
            "Final test error",
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
