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

def run_one_experiment(experiment, dataset, params, n_repeats):
    """ Run one experiment, for a given set of parameters, and for a given
    number of repeats. This function is called by run_all_experiments in a loop.
    """
    # Initialise the list of results for this experiment
    experiment_results = []
    # Run experiment and store the results
    for i in range(n_repeats):
        np.random.seed(i)
        try:
            result = experiment(dataset, **params)
            test_error_column = optimisers.results.columns.TestError
            test_errors = result.get_values(test_error_column)
            experiment_results.append(test_errors[-1])
        except:
            print_error_details(params)
    
    # Return the results
    return experiment_results

def get_best_param(experiment_results, n_sigma=1):
    """ Given a dictionary experiment_results, which maps values of a parameter
    to a list of results, return the parameter which minimises a linear
    combination of the mean and the standard deviation of the corresponding list
    of results, where the relative contribution of the standard deviation is
    given by the argument n_sigma (if large and positive, then a lower standard
    deviation is considered to give a relatively better parameter value) """
    # Make dictionary mapping values to scores
    score_dict = {
        val: np.mean(results) + n_sigma * np.std(results)
        for val, results in experiment_results.items()
    }
    # Find the best score
    best_score = min(score_dict.values())
    # Find the list of values that have the best score
    best_value_list = [
        val for val in experiment_results if score_dict[val] == best_score
    ]
    # Return the first value with the best score
    return best_value_list[0]

def dict_to_tuple(d):
    """ Convert a dictionary to a tuple (this tuple can then be used as the key
    for a dictionary, because it is hashable) """
    return tuple((k, v) for k, v in d.items())

def run_all_experiments(
    all_experiments_dict,
    experiment,
    dataset,
    output_dir,
    n_repeats=5,
    verbose=True,
    find_best_parameters=False
):
    """
    Run experiments for each parameter specified in all_experiments_dict, and
    save plots for each parameter under test to disk. Optionally also find the
    best value for each parameter under test.

    Inputs:
    -   all_experiments_dict: dictionary in which each key is the name of a
        parameter that will be passed to experiment, and each value is a
        dictionary containing a "default" key and a "range" key, for the default
        value to use when testing other parameters, and the list of values to
        try when testing this parameter. Note that the default parameters for
        each parameter may be modified in place by this function
    -   experiment: a callable that should accept a Dataset object as the first
        positional argument and all of the keys in all_experiments_dict as
        keyword arguments, run the corresponding experiment, and return a Result
        object
    -   dataset: a Dataset object to pass to experiment
    -   output_dir: directory in which to save plots for each parameter under
        test
    -   n_repeats: number of repeats for each parameter combination
    -   verbose: whether to print out each time a plot is being made, and total
        time taken
    -   find_best_parameters: if True, then iteratively loop through this
        function, updating the default parameters to the best parameter values
        found from each experiment, until the approximately locally optimal
        parameter values have been found for each parameter under test

    Outputs:
    -   None
    -   (plots for each parameter under test are saved to disk)
    """
    t_0 = perf_counter()

    # Initialise dictionary mapping experiment parameters to results
    param_dict_to_result_dict = {}
    while True:
        run_any_experiments = False
        # Iterate through each parameter that will be varied
        for var_param_name, var_param_dict in all_experiments_dict.items():
            # Initialise parameter dictionary using defaults
            experiment_params = {
                param_name: param_dict["default"]
                for param_name, param_dict
                in all_experiments_dict.items()
            }
            # Initialise results dictionary that will be used for plotting
            experiment_results = {}

            # Iterate through each value of the parameter under test
            for var_param_value in var_param_dict["range"]:
                # Set the value of the parameter under test for this experiment
                experiment_params[var_param_name] = var_param_value
                # Check if this experiment has already been run
                experiment_tup = dict_to_tuple(experiment_params)
                if experiment_tup not in param_dict_to_result_dict:
                    # Run the experiment, store the results
                    results_list = run_one_experiment(
                        experiment,
                        dataset,
                        experiment_params,
                        n_repeats
                    )
                    experiment_results[var_param_value] = results_list
                    param_dict_to_result_dict[experiment_tup] = results_list
                    run_any_experiments = True
                else:
                    # This experiment has been run, so retrieve the results
                    results_list = param_dict_to_result_dict[experiment_tup]
                    experiment_results[var_param_value] = results_list
            
            if find_best_parameters:
                # Update the default for the parameter under test
                best_param = get_best_param(experiment_results)
                all_experiments_dict[var_param_name]["default"] = best_param

            if run_any_experiments:
                if verbose:
                    if find_best_parameters:
                        print("New default value for %r is %r" % (
                            var_param_name,
                            all_experiments_dict[var_param_name]["default"]
                        ))
                    msg = "Plotting result for {}".format(var_param_name)
                    h_line = "*" * len(msg)
                    print("", h_line, msg, h_line, "", sep="\n")
                # Test if all x-values are numeric
                all_numeric = all(
                    isinstance(x, int) or isinstance(x, float)
                    for x in experiment_results.keys()
                )
                # If not all x-values are numeric, then format as strings
                if not all_numeric:
                    experiment_results = {
                        str(val).replace("activation function", "").rstrip():
                        e_list
                        for val, e_list in experiment_results.items()
                    }
                # Call plotting function
                plotting.plot_parameter_experiment_results(
                    experiment_results,
                    var_param_name,
                    "Varying parameter {}".format(var_param_name),
                    output_dir,
                )

        # Break if not finding best parameters, or after running no experiments
        if (not find_best_parameters) or (not run_any_experiments):
            break

    if verbose:
        if find_best_parameters:
            # Display best parameters
            print("Best parameters found:")
            print(
                "\n".join(
                    "%s: %s" % (var_param_name, var_param_dict["default"])
                    for (var_param_name, var_param_dict)
                    in all_experiments_dict.items()
                ),
                end="\n\n"
            )

        # Print total time taken to run all experiments
        mins, secs = divmod(perf_counter() - t_0, 60)
        if mins > 0:
            print("All tests run in %i mins %.2f s" % (mins, secs))
        else:
            print("All tests run in %.2f s" % secs)
