"""
Module containing the test-neutral run_all_experiments function for comparing
parameters. This function should be called by wrapper scripts for particular
optimisers and sets of parameters to test.

TODO:
-   refactor the run_all_experiments function and its subfunctions into classes,
    EG a Parameter class (containing attributes name, default, and range, and a
    __repr__ method) and an Experiment class (containing methods add_parameter,
    sweep_parameter, sweep_all_parameters, find_best_parameters*, etc, and a
    custom exception for calling the Experiment's function with missing/extra
    Parameters). Then, once the training module is tidied up, move this module
    into the training module, and add unit tests (including tests that the
    correct exception is raised when missing/extra arguments are given).
-   A unit test should be added with a few different parameters, a dataset =
    None, and a function that returns the sum of the squared parameters, with
    initially non-zero defaults, and check that calling find_best_parameters
    returns all zero default parameters
-   Test that calling the sweep_all_parameters method (not from within the
    find_best_parameters method) leaves all parameter default values unchanged

* the class should initialise with a finding_best_parameters attribute which is
  set to False. When find_best_parameters is called, finding_best_parameters
  should be set to True at the start of the function, and reset to False at the
  end of the function.
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

    TODO: print to an "Error logs" subdirectory of the Experiment output_dir
    attribute. output_dir should be made in the Experiment initialiser. This
    function should become a private method
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
            # TODO: the scalar test_errors[-1] should be returned by the
            # experiment function
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
    return tuple(sorted(list(d.items())))

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
            # Initialise results dictionary that will be used for plotting TODO:
            # rename this param_sweep_results
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

    # Save dictionary mapping parameters to results to disk
    with open(os.path.join(output_dir, "Results.txt"), "w") as f:
        for tup, results_list in param_dict_to_result_dict.items():
            print("%s\n\t%s\n" % (tup, results_list), file=f)

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

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

class Parameter:
    """ Class to represent parameters that are swept over by the Experiment
    class """
    def __init__(self, name, default, val_range):
        """ Initialise a Parameter object.

        Inputs:
        -   name: string used to refer to the parameter, and as a key in the
            keyword argument dictionary passed to the Experiments function
        -   default: value passed to the Experiment's function when this
            parameter is not being swept over
        -   val_range: the range of values for this parameter that will be swept
            over during an experiment
        """
        self.name = name
        self.default = default
        self.val_range = val_range
    
    def __repr__(self):
        """ Return a string representation of this object """
        s = "Parameter(%r, %r, %r)" % (self.name, self.default, self.val_range)
        return s

class Experiment:
    """ Class which is used to perform abstract experiments to sweep over
    parameters, plot graphs of the results, and also find approximately locally
    optimal parameters """
    def __init__(self, func, output_dir, n_repeats=5, n_sigma=1, verbose=True):
        """ Initialise an Experiment object.

        Inputs:
        -   func: callable which accepts a data.Dataset object and several
            keyword arguments, whose names correspond to the names of the
            Parameter objects that are added to this object, and returns a score
            (lower is better), EG the final test-set error
        -   
        """
        self._func = func
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        self._output_dir = output_dir
        self._n_repeats = n_repeats
        self._n_sigma = n_sigma
        self._verbose = True
        self._param_list = []
        self._params_to_results_dict = dict()
        self._has_run_any_experiments = False
    
    def add_parameter(self, parameter):
        """ Add a parameter to the internal list of parameters. parameter should
        be an instance of Parameter. The values of  """
        self._param_list.append(parameter)
    
    def sweep_parameter(self, parameter, plot=True, update_parameters=False):
        """ TODO: add verbosity and printing """
        experiment_params = self._get_default_dictionary()
        # Initialise dictionary mapping parameter values to results lists
        param_sweep_results = {}

        # Iterate through each value of the parameter under test
        for val in parameter.val_range:
            # Set the value of the parameter under test for this experiment
            experiment_params[parameter.name] = val
            # Check if this experiment has already been run
            if not self._has_experiment_results(experiment_params):
                # Run the experiment, store the results
                results_list = self._run_experiment(experiment_params)
                param_sweep_results[val] = results_list
                self._set_experiment_results(experiment_params, results_list)
                self._has_run_any_experiments = True
            else:
                # This experiment has been run, so retrieve the results
                results_list = self._get_experiment_results(experiment_params)
                param_sweep_results[val] = results_list
        
        if update_parameters:
            # Update the default for the parameter under test
            old_default = parameter.default
            best_param = self._get_best_param(param_sweep_results)
            parameter.default = best_param
            if self._verbose and (old_default != best_param):
                print("...")
        
        if plot:
            if self._verbose:
                print("...")
            # Call plotting function
            plotting.plot_parameter_experiment_results(
                param_sweep_results,
                parameter.name,
                "Varying parameter %s" % parameter.name,
                self._output_dir,
                self._n_sigma
            )

        # Return the results
        return param_sweep_results

    def sweep_all_parameters(self, plot=True, update_parameters=False):
        """  """
        for parameter in self._param_list:
            self.sweep_parameter(parameter, plot, update_parameters)
    
    def find_best_parameters(self, plot=True):
        """  """
        while True:
            self._has_run_any_experiments = False
            self.sweep_all_parameters(plot, True)
            if not self._has_run_any_experiments:
                break

    def save_results_as_text(self):
        """ Save each set of parameters which has been tested (along with the
        corresponding list of results) to disk as a text file """
        with open(os.path.join(self._output_dir, "Results.txt"), "w") as f:
            for tup, results_list in self._params_to_results_dict.items():
                # print("%s\n\t%s\n" % (tup, results_list), file=f)
                print(", ".join("%s = %s" % (k, v) for (k, v) in tup))
                print("\t%s\n" % results_list)

    def _get_default_dictionary(self):
        """ Return a dictionary in which the keys are the names of each of the
        Parameter objects which have been added to this Experiment object, and
        the values are the default values for each of those Parameter objects
        """
        return {param.name: param.default for param in self._param_list}

    def _has_experiment_results(self, experiment_params):
        """  """
        experiment_tup = dict_to_tuple(experiment_params)
        return (experiment_tup in self._params_to_results_dict)

    def _get_experiment_results(self, experiment_params):
        """  """
        experiment_tup = dict_to_tuple(experiment_params)
        return self._params_to_results_dict[experiment_tup]

    def _set_experiment_results(self, experiment_params, results_list):
        """  """
        experiment_tup = dict_to_tuple(experiment_params)
        self._params_to_results_dict[experiment_tup] = results_list

    def _run_experiment(self, experiment_params):
        """ Run an experiment with a given set of parameters, and return the
        list of scores for each repeat of the experiment. This method is called
        by the sweep_parameter method in a loop """
        # Initialise the list of results for this experiment
        results_list = []
        # Run experiment and store the results
        for i in range(self._n_repeats):
            np.random.seed(i)
            try:
                score = self._func(**experiment_params)
                results_list.append(score)
            except:
                self._print_exception_details(experiment_params)
        
        # Return the results
        return results_list
    
    def _print_exception_details(self, experiment_params):
        """ Print details of an exception both to STDOUT and to a time-stamped
        error log file """
        # Get the name of the error directory, and make it if it doesn't exist
        error_dir = os.path.join(self._output_dir, "Error logs")
        if not os.path.isdir(error_dir):
            os.makedirs(error_dir)
        # Get the full path to the error log file
        t = datetime.now()
        error_filename = "{} Error log.txt".format(t).replace(":", ".")
        error_file_path = os.path.join(error_dir, error_filename)
        # Open the file
        with open(error_file_path, "w") as error_file:
            # Print to the text file, and to STDOUT
            for f in [error_file, None]:
                # Print the parameter details
                print("Parameter details:", file=f)
                for key, value in experiment_params.items():
                    print("{}: {}".format(repr(key), value), file=f)
                print("*" * 50, file=f)
                # Print the exception details
                print("Error details:", file=f)
                print_exception(*sys.exc_info(), file=f)

    def _get_best_param(self, experiment_results):
        """ Given a dictionary experiment_results, which maps values of a
        parameter to a list of results, return the parameter which minimises a
        linear combination of the mean and the standard deviation of the
        corresponding list of results, where the relative contribution of the
        standard deviation is given by self._n_sigma (if large and positive,
        then a lower standard deviation is considered to give a relatively
        better parameter value) """
        # Make dictionary mapping values to scores
        score_dict = {
            val: (np.mean(results) + self._n_sigma * np.std(results))
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
