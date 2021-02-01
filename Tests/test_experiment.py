import os
import string
import numpy as np
import pytest
from experiment import Experiment, Parameter
from .util import get_dataset, dataset_list, get_output_dir

# Get name of output directory, and create it if it doesn't exist
output_dir = get_output_dir("Experiment")

@pytest.mark.parametrize(
    "seed, plot",
    [(491, True), (6940, False), (2903, False)]
)
def test_find_best_parameters(seed, plot):
    """ Test that the Experiment.find_best_parameters method does indeed find
    the best parameters. As an objective function we use the norm of the inputs,
    for which the best parameters are the ones which have the lowest absolute
    value, and check that the best values are found. Also test plotting the best
    parameters """
    # Set the random seed
    np.random.seed(seed)
    # Set the number of parameters
    if plot:
        # If plotting, only use a small number of parameters to save time
        num_params = 3
    else:
        # Otherwise use a larger random number of parameters
        num_params = np.random.randint(5, 10)

    # Define the function that will be called by the Experiment object
    def run_experiment(**kwargs):
        """ Function that will be called by the Experiment object """
        # Check that we have the right number of inputs
        assert len(kwargs) == num_params
        # Return the norm of the input values
        return np.linalg.norm(list(kwargs.values()))
    
    # Get the output directory name, and create it if it doesn't exist already
    test_output_dir = os.path.join(
        output_dir,
        "test_find_best_parameters, seed = %i" % seed
    )
    if not os.path.isdir(test_output_dir):
        os.makedirs(test_output_dir)
    # Get the output filename and open it
    output_filename = os.path.join(test_output_dir, "output.txt")
    with open(output_filename, "w") as f:
        # Initialise the experiment object
        experiment = Experiment(run_experiment, test_output_dir, output_file=f)
        
        # Define shortcut expression for adding parameters
        addp = lambda *args: experiment.add_parameter(Parameter(*args))
        # Iterate through each parameter
        for i in range(num_params):
            # Choose a unique and valid parameter name
            name = string.ascii_letters[i]
            # Choose a random number of values, range of values, and default
            num_values = np.random.randint(5, 10)
            val_range = np.random.normal(size=num_values)
            default = np.random.choice(val_range)
            # Add the parameter to the Experiment object
            addp(name, default, val_range)

        # Call the method to find the best parameter values
        experiment.find_best_parameters(plot)

    # Iterate through each parameter in the Experiment object
    for param in experiment._param_list:
        # Find the minimum absolute parameter value
        min_abs_val = min(abs(x) for x in param.val_range)
        # Check that the default parameter value is the best one
        assert abs(param.default) == min_abs_val

@pytest.mark.parametrize("seed", [4061, 44, 8589])
def test_sweep_all_parameters(seed):
    """ Test calling the Experiment.sweep_all_parameters method, including that
    the parameters are not updated unless specified """
    # Set the random seed
    np.random.seed(seed)
    # Set the number of parameters
    num_params = np.random.randint(5, 10)

    # Define the function that will be called by the Experiment object
    def run_experiment(**kwargs):
        """ Function that will be called by the Experiment object """
        # Check that we have the right number of inputs
        assert len(kwargs) == num_params
        # Return the norm of the input values
        return np.linalg.norm(list(kwargs.values()))
    
    # Get the output directory name, and create it if it doesn't exist already
    test_output_dir = os.path.join(
        output_dir,
        "test_sweep_all_parameters, seed = %i" % seed
    )
    if not os.path.isdir(test_output_dir):
        os.makedirs(test_output_dir)
    # Get the output filename and open it
    output_filename = os.path.join(test_output_dir, "output.txt")
    with open(output_filename, "w") as f:
        # Initialise the experiment object
        experiment = Experiment(run_experiment, test_output_dir, output_file=f)
        
        # Define shortcut expression for adding parameters
        addp = lambda *args: experiment.add_parameter(Parameter(*args))
        # Iterate through each parameter
        for i in range(num_params):
            # Choose a unique and valid parameter name
            name = string.ascii_letters[i]
            # Choose a random number of values and range of values
            num_values = np.random.randint(5, 10)
            val_range = np.random.normal(size=num_values)
            # Set the default value to the worst possible value
            max_abs_val = max(abs(x) for x in val_range)
            default = [val for val in val_range if abs(val) == max_abs_val][0]
            # Add the parameter to the Experiment object
            addp(name, default, val_range)

        # Get dictionary of original parameter names and defaults
        param_defaults_original = experiment._get_default_dictionary()
        # Call the method to sweep over all the parameter values
        f.write("Calling sweep_all_parameters with update_parameters=False\n")
        experiment.sweep_all_parameters(plot=False, update_parameters=False)
        # Get dictionary of updated parameter names and defaults
        param_defaults_new = experiment._get_default_dictionary()
        # Assert that the parameter defaults haven't changed
        assert param_defaults_new == param_defaults_original
        
        # Call the same method, but this time also update the parameters
        f.write("Calling sweep_all_parameters with update_parameters=True\n")
        experiment.sweep_all_parameters(plot=False, update_parameters=True)
        # Get dictionary of updated parameter names and defaults
        param_defaults_new = experiment._get_default_dictionary()
        # Assert that this time the parameter defaults HAVE changed
        assert param_defaults_new != param_defaults_original
