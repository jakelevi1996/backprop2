"""
Module containing unit tests for the activations module.

TODO: write decorator in util module to parameterise unit tests with different
error functions, instead of repeated for-loops?
"""
import os
import numpy as np
import errors as e
from .util import get_random_network_inputs_targets, iterate_random_seeds

# Define list of activation functions to be tested
error_func_list = [e.SumOfSquares()]

# Get name of output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs")

@iterate_random_seeds(2144, 6646, 9914)
def test_shapes():
    """
    Test that the output from each activation function and its derivatives are
    the correct shapes

    TODO: second derivatives
    """
    for error_func in error_func_list:
        n, x, t, N_D = get_random_network_inputs_targets()
        y = n.forward_prop(x)
        assert error_func(y, t).shape == (1, N_D)
        assert error_func.dEdy(y, t).shape == (n.output_dim, N_D)

@iterate_random_seeds(3995, 1218, 589)
def test_propagation():
    """
    Test that the activation function can be used for back propagation and
    calculating the mean error in a neural network model
    """
    for error_func in error_func_list:
        n, x, t, _ = get_random_network_inputs_targets(error_func=error_func)
        n.back_prop(x, t)
        n.mean_error(t, x)

@iterate_random_seeds(3832, 4226, 1727)
def test_act_func_id():
    """
    Test that the activation function IDs are consistent
    """
    # Check that you get the correct activation function back from its id
    for error_func in error_func_list:
        error_func_id = error_func.get_id_from_func()
        error_func_from_id = e.ErrorFunction().get_func_from_id(error_func_id)
        assert type(error_func_from_id) is type(error_func)
    
    # Check that all of the ids are unique
    id_list = [error_func.get_id_from_func() for error_func in error_func_list]
    assert len(set(id_list)) == len(id_list)

@iterate_random_seeds(6309, 7639, 2532)
def test_plotting():
    for error_func in error_func_list:
        error_func.plot(output_dir)
