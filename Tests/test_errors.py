"""
Module containing unit tests for the errors module.
"""
import os
import pytest
import numpy as np
import errors as e
from .util import get_random_network_inputs_targets, iterate_random_seeds

# Define list of error functions to be tested
error_func_list = [e.SumOfSquares()]

# Get name of output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs")

@pytest.mark.parametrize("seed", [2144, 6646, 9914])
@pytest.mark.parametrize("error_func", error_func_list)
def test_shapes(seed, error_func):
    """
    Test that the output from each error function and its derivatives are
    the correct shapes

    TODO: second derivatives
    """
    n, x, t, N_D = get_random_network_inputs_targets(seed)
    y = n.forward_prop(x)
    assert error_func(y, t).shape == (1, N_D)
    assert error_func.dEdy(y, t).shape == (n.output_dim, N_D)

@pytest.mark.parametrize("seed", [3995, 1218, 589])
@pytest.mark.parametrize("error_func", error_func_list)
def test_propagation(seed, error_func):
    """
    Test that the error function can be used for back propagation and
    calculating the mean error in a neural network model
    """
    n, x, t, _ = get_random_network_inputs_targets(seed, error_func=error_func)
    n.back_prop(x, t)
    n.mean_error(t, x)

@pytest.mark.parametrize("seed", [5331, 3475, 9941])
@pytest.mark.parametrize("error_func", error_func_list)
def test_error_func_id(seed, error_func):
    """
    Test that the error function IDs are self-consistent, IE that the correct
    error function is returned from its id
    """
    # Check that you get the correct error function back from its id
    error_func_id = error_func.get_id_from_func()
    error_func_from_id = e.get_func_from_id(error_func_id)
    assert type(error_func_from_id) is type(error_func)
    # Check that the outputs are consisitent
    _, x, _, _ = get_random_network_inputs_targets(seed)
    assert np.all(error_func_from_id(x, 0) == error_func(x, 0))
    
def test_error_func_ids_unique():
    """ Check that all of the error function IDs are unique """
    # Check that all of the ids are unique
    id_list = [error_func.get_id_from_func() for error_func in error_func_list]
    assert len(set(id_list)) == len(id_list)

@pytest.mark.parametrize("error_func", error_func_list)
def test_plotting(error_func):
    error_func.plot(output_dir)
