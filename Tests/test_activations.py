"""
Module containing unit tests for the activations module.

TODO: write decorator in util module to parameterise unit tests with different
activation functions, instead of repeated for-loops?
"""
import os
import numpy as np
import activations as a
from .util import get_random_network_inputs_targets, iterate_random_seeds

# Define list of activation functions to be tested
act_func_list = [
    a.Identity(), a.Logistic(), a.Relu(), a.Gaussian()
]

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
    for act_func in act_func_list:
        _, x, _, _ = get_random_network_inputs_targets()
        assert act_func(x).shape == x.shape
        assert act_func.dydx(x).shape == x.shape

@iterate_random_seeds(3995, 1218, 589)
def test_propagation():
    """
    Test that the activation function can be used for forward and back
    propagation in a neural network model
    """
    for act_func in act_func_list:
        n, x, t, _ = get_random_network_inputs_targets(act_funcs=[act_func])
        n.forward_prop(x)
        n.back_prop(x, t)

@iterate_random_seeds(3832, 4226, 1727)
def test_act_func_id():
    """
    Test that the activation function IDs are consistent
    """
    # Check that you get the correct activation function back from its id
    for act_func in act_func_list:
        act_func_id = act_func.get_id_from_func()
        act_func_from_id = a.ActivationFunction().get_func_from_id(act_func_id)
        assert type(act_func_from_id) is type(act_func)
    
    # Check that all of the ids are unique
    id_list = [act_func.get_id_from_func() for act_func in act_func_list]
    assert len(set(id_list)) == len(id_list)

@iterate_random_seeds(6309, 7639, 2532)
def test_plotting():
    for act_func in act_func_list:
        act_func.plot(output_dir)
