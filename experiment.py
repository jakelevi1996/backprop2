""" Module containing the Experiment and Parameter classes, used to perform
abstract experiments to sweep over parameters, plot graphs of the results, and
also find approximately locally optimal parameters. """
import os
import sys
from datetime import datetime
from traceback import print_exception
import numpy as np
if __name__ == "__main__":
    import __init__
import plotting

def dict_to_tuple(d):
    """ Convert a dictionary to a tuple (this tuple can then be used as the key
    for a dictionary, because it is hashable) """
    return tuple(sorted(d.items()))

def set_seed(i, experiment_params):
    """ Given the iteration number i, and a dictionary of parameters for an
    experiment, set an almost-surely-unique random seed for the experiment. The
    input to np.random.seed "must be between 0 and 2**32 - 1", hence a 32-bit
    mask is applied to the seed """
    experiment_descriptor = (i, dict_to_tuple(experiment_params))
    mask = (1 << 32) - 1
    seed = hash(experiment_descriptor) & mask
    np.random.seed(seed)

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
    def __init__(
        self,
        func,
        output_dir,
        n_repeats=5,
        n_sigma=1,
        output_file=None
    ):
        """ Initialise an Experiment object.

        Inputs:
        -   func: callable which accepts arguments whose names correspond to the
            names of the Parameter objects that will be added to this object,
            and returns a score (lower is better, EG the final test-set error)
        -   output_dir: directory in which to plot results, store errors logs
        -   n_repeats: number of times to repeat each experiment
        -   n_sigma: relative importance of standard deviation when deciding the
            best parameter value (if large and positive, then a lower standard
            deviation is considered to give a relatively better parameter value)
        -   output_file: the file to print logging output to. Default is None,
            in which case print to stdout
        """
        self._func = func
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        self._output_dir = output_dir
        self._n_repeats = n_repeats
        self._n_sigma = n_sigma
        self._file = output_file
        self._param_list = list()
        self._params_to_results_dict = dict()
        self._has_updated_any_parameters = False
    
    def add_parameter(self, parameter):
        """ Add a parameter to the internal list of parameters. parameter should
        be an instance of Parameter """
        self._param_list.append(parameter)
    
    def sweep_parameter(self, parameter, plot=True, update_parameters=False):
        """ Given a Parameter object (that should have already been added to
        this Experiment object using the add_parameter method), iterate over all
        of the values in the val_range attribute of this parameter, and run one
        experiment (with multiple repeats) for each of these values, with each
        of the other parameters taking their default values.

        This method can be called as a public method, and is also called by the
        sweep_all_parameters public method.

        Inputs:
        -   parameter: Parameter object whose val_range attribute will be
            iterated over
        -   plot: if True then plot the results from sweeping over this
            parameter
        -   update_parameters: if True then update the default value of this
            parameter to the value that gives the best score (see
            _get_best_param_val method)
        """
        # Initialise dictionary mapping parameter names to their default values
        experiment_params = self._get_default_dictionary()
        # Initialise dictionary mapping parameter values to results lists
        param_sweep_results = dict()

        # Iterate through each specified value of the parameter under test
        for val in parameter.val_range:
            # Set the value of the parameter under test for this experiment
            experiment_params[parameter.name] = val
            # Check if this experiment has already been run
            if not self._has_experiment_results(experiment_params):
                # Run the experiment and store the results
                results_list = self._run_experiment(experiment_params)
                self._set_experiment_results(experiment_params, results_list)
            else:
                # This experiment has already been run, so retrieve the results
                results_list = self._get_experiment_results(experiment_params)
            
            # Store the results, ready for updating parameters and plotting
            param_sweep_results[val] = results_list
        
        if update_parameters:
            # Check if the default for this parameter has the best value
            best_param_val = self._get_best_param_val(param_sweep_results)
            if parameter.default != best_param_val:
                self._print(
                    "Parameter %r default value changing from %s to %s"
                    % (parameter.name, parameter.default, best_param_val),
                )
                # Update the default for the parameter under test
                parameter.default = best_param_val
                self._has_updated_any_parameters = True
        
        if plot:
            self._print("Plotting results for parameter %r..." % parameter.name)
            # Format the dictionary keys
            param_sweep_results = self._format_result_keys(param_sweep_results)
            # Call plotting function
            plotting.plot_parameter_sweep_results(
                param_sweep_results,
                parameter.name,
                "Varying parameter %s" % parameter.name,
                self._output_dir,
                self._n_sigma
            )

        # Return the results
        return param_sweep_results

    def sweep_all_parameters(self, plot=True, update_parameters=False):
        """ Iterate through each parameter that has been added to this
        Experiment object, and call the sweep_parameter method once with each
        parameter as the parameter-under-test.

        This method can be called as a public method, and is also called by the
        find_best_parameters public method. """
        for parameter in self._param_list:
            self._print("Sweeping over parameter %r..." % parameter.name)
            self.sweep_parameter(parameter, plot, update_parameters)
    
    def find_best_parameters(self, plot=True):
        """ Iteratively call the sweep_all_parameters method, updating the
        default value for each parameter each time it is swept over to the value
        which gives the best score, until we have an iteration in which no
        parameters are updated; at this point we stop iterating, because we know
        that more experiments will not change the results or default value for
        any parameters, and therefore that we have found the approximately
        locally optimal value for each parameter (based on the parameter default
        and val_range attributes, and the finite number of repeats used in each
        experiment) """
        # Iterate until an iteration in which no parameters are updated
        while True:
            # At the start of each iteration, no parameters have been updated
            self._has_updated_any_parameters = False
            # Sweep over all parameters, updating defaults to the best values
            self.sweep_all_parameters(plot=False, update_parameters=True)
            # Check if any parameter defaults were updated
            if not self._has_updated_any_parameters:
                # No parameters were updated this iteration, so stop
                self._print("Finished sweeping through parameters")
                break
        
        if plot:
            self._print("Plotting results...")
            # Plot the results for each parameter
            self.sweep_all_parameters(plot=True, update_parameters=True)
        
        # Print the best parameters found
        self._print(
            "Best parameters found:\n" +
            "\n".join(
                "%r: %s" % (param.name, param.default)
                for param in self._param_list
            )
        )

    def save_results_as_text(self):
        """ Save each set of parameters which has been tested (along with the
        corresponding list of results) and the final default parameter values to
        disk as a text file """
        with open(os.path.join(self._output_dir, "Results.txt"), "w") as f:
            # Iterate through each experiment that was run
            for tup, results_list in self._params_to_results_dict.items():
                # Print the parameters for the experiment
                print(", ".join("%r = %s" % (k, v) for (k, v) in tup), file=f)
                # Print the results for the experiment
                print("\t%s\n" % results_list, file=f)
            # Print the final default parameter values
            print(
                "\n\nFinal default parameter values:",
                "\n".join(
                    "%r: %s" % (param.name, param.default)
                    for param in self._param_list
                ),
                sep="\n",
                file=f
            )

    def _get_default_dictionary(self):
        """ Return a dictionary in which the keys are the names of each of the
        Parameter objects which have been added to this Experiment object, and
        the values are the default values for each of those Parameter objects
        """
        return {param.name: param.default for param in self._param_list}

    def _has_experiment_results(self, experiment_params):
        """ Check if an experiment has already been run with the given
        dictionary of parameters """
        experiment_tup = dict_to_tuple(experiment_params)
        return (experiment_tup in self._params_to_results_dict)

    def _get_experiment_results(self, experiment_params):
        """ Get the results from when an experiment was run with the given
        dictionary of parameters """
        experiment_tup = dict_to_tuple(experiment_params)
        return self._params_to_results_dict[experiment_tup]

    def _set_experiment_results(self, experiment_params, results_list):
        """ Store the results from running an experiment with the given
        dictionary of parameters """
        experiment_tup = dict_to_tuple(experiment_params)
        self._params_to_results_dict[experiment_tup] = results_list

    def _run_experiment(self, experiment_params):
        """ Run an experiment with a given dictionary of parameters, repeated
        several times with different random seeds, and return the list of scores
        for each repeat of the experiment. This method is called by the
        sweep_parameter method in a loop """
        self._print(
            "Running an experiment with parameters: " +
            ", ".join(
                "%r = %s" % (name, value)
                for (name, value) in experiment_params.items()
            )
        )
        # Initialise the list of results for this experiment
        results_list = []
        # Run experiment and store the results
        for i in range(self._n_repeats):
            set_seed(i, experiment_params)
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
                    print("%r: %s" % (key, value), file=f)
                print("*" * 50, file=f)
                # Print the exception details
                print("Error details:", file=f)
                print_exception(*sys.exc_info(), file=f)

    def _get_best_param_val(self, experiment_results):
        """ Given a dictionary experiment_results, which maps values of a
        parameter to a list of results, return the parameter value which
        minimises a linear combination of the mean and the standard deviation of
        the corresponding list of results; the relative contribution of the
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
            val for val in experiment_results
            if score_dict[val] == best_score
        ]
        # Return the first value with the best score
        return best_value_list[0]

    def _print(self, s):
        """ Print the string s to the output file for this Experiment object, or
        to stdout if self._file is None """
        print(s, file=self._file)

    def _format_result_keys(self, param_sweep_results):
        """ Given a dictionary of results from the sweep_parameter method,
        format the dictionary keys so that they will look nice when plotted
        (these dictionary keys will be used as the x-values) """
        # Test if all dictionary keys are numeric
        all_numeric = all(
            isinstance(k, int) or isinstance(k, float)
            for k in param_sweep_results.keys()
        )
        # If not all dictionary keys are numeric, then format as strings
        if not all_numeric:
            param_sweep_results = {
                str(key).replace("activation function", "").rstrip(): value
                for key, value in param_sweep_results.items()
            }
        
        # Return the dictionary
        return param_sweep_results
