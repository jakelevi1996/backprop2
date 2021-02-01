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
    value, and check that the best values are found """
    np.random.seed(seed)
    if plot:
        num_params = 3
    else:
        num_params = np.random.randint(5, 10)
    param_names = list(string.ascii_letters[:num_params])
    def run_experiment(**kwargs):
        assert len(kwargs) == num_params
        return np.linalg.norm(list(kwargs.values()))
    
    test_output_dir = os.path.join(
        output_dir,
        "test_find_best_parameters, seed = %i" % seed
    )
    if not os.path.isdir(test_output_dir):
        os.makedirs(test_output_dir)
    output_filename = os.path.join(test_output_dir, "output.txt")
    f = open(output_filename, "w")
    experiment = Experiment(run_experiment, test_output_dir, output_file=f)
    
    addp = lambda *args: experiment.add_parameter(Parameter(*args))

    for i in range(num_params):
        name = string.ascii_letters[i]
        num_values = np.random.randint(5, 10)
        val_range = np.random.normal(size=num_values)
        default = np.random.choice(val_range)
        addp(name, default, val_range)

    experiment.find_best_parameters(plot)

    f.close()

    for param in experiment._param_list:
        # Find the minimum absolute parameter value
        min_abs_val = min(abs(x) for x in param.val_range)
        # Check that the default parameter value is the best one
        assert abs(param.default) == min_abs_val
