""" Module containing unit tests for the errors module. """
import os
import pytest
import numpy as np
import models
from .util import (
    get_output_dir,
    set_random_seed_from_args,
    get_random_network_inputs_targets,
)

# Define list of error functions to be tested
error_func_list = [
    models.errors.sum_of_squares,
    models.errors.softmax_cross_entropy,
    models.errors.binary_cross_entropy,
]

# Get name of output directory, and create it if it doesn't exist
output_dir = get_output_dir("Errors")

def _get_output_dim_act_funcs(error_func):
    """ Get the output dimension and activation functions, given the error
    function. This is because the binary cross-entropy error function requires
    one-dimensional outputs in [0, 1] (so a logistic activation function is
    used in the final layer), whereas for the other error functions, any output
    dimension and activation functions should be valid, so None is returned,
    leading to defaults being used in the functions to which they're passed """
    if (error_func is models.errors.binary_cross_entropy):
        output_dim = 1
        act_funcs = [models.activations.logistic]
    else:
        output_dim = None
        act_funcs = None

    return output_dim, act_funcs


@pytest.mark.parametrize("repeat", range(3))
@pytest.mark.parametrize("error_func", error_func_list)
def test_shapes(repeat, error_func):
    """ Test that the output from each error function and its derivatives are
    the correct shapes """
    set_random_seed_from_args("test_shapes", repeat)
    output_dim, act_funcs = _get_output_dim_act_funcs(error_func)
    n, x, t, N_D = get_random_network_inputs_targets(
        output_dim=output_dim,
        act_funcs=act_funcs,
    )
    y = n.forward_prop(x)
    assert error_func(y, t).shape == (1, N_D)
    assert error_func.dEdy(y, t).shape == (n.output_dim, N_D)
    assert error_func.d2Edy2(y, t).shape == (n.output_dim, n.output_dim, N_D)

@pytest.mark.parametrize("repeat", range(3))
@pytest.mark.parametrize("error_func", error_func_list)
def test_propagation(repeat, error_func):
    """ Test that the error function can be used for back propagation and
    calculating the mean error in a neural network model """
    output_dim, act_funcs = _get_output_dim_act_funcs(error_func)
    set_random_seed_from_args("test_propagation", repeat)
    n, x, t, _ = get_random_network_inputs_targets(
        output_dim=output_dim,
        act_funcs=act_funcs,
        error_func=error_func,
    )
    n.forward_prop(x)
    n.back_prop(x, t)
    n.back_prop2(x, t)
    n.mean_total_error(t)

@pytest.mark.parametrize("repeat", range(3))
@pytest.mark.parametrize("error_func", error_func_list)
def test_error_func_id(repeat, error_func):
    """ Test that the error function IDs are self-consistent, IE that the
    correct error function is returned from its id """
    # Check that you get the correct error function back from its id
    error_func_id = error_func.get_id_from_func()
    error_func_from_id = models.errors.get_func_from_id(error_func_id)
    assert type(error_func_from_id) is type(error_func)
    # Check that the outputs are consisitent
    output_dim, act_funcs = _get_output_dim_act_funcs(error_func)
    set_random_seed_from_args("test_error_func_id", repeat)
    n, x, _, _ = get_random_network_inputs_targets(
        output_dim=output_dim,
        act_funcs=act_funcs,
    )
    y = n(x)
    assert np.all(error_func_from_id(y, 0) == error_func(y, 0))

def test_error_func_ids_unique():
    """ Check that all of the error function IDs are unique """
    id_list = [error_func.get_id_from_func() for error_func in error_func_list]
    assert len(set(id_list)) == len(id_list)

@pytest.mark.parametrize("error_func", error_func_list)
def test_plotting(error_func):
    error_func.plot(output_dir)

def test_id_from_invalid_error_func():
    """ Test that trying to get an id from the abstract parent class
    _ErrorFunction raises a RuntimeError """
    with pytest.raises(RuntimeError):
        models.errors._ErrorFunction().get_id_from_func()
