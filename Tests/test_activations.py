"""
Module containing unit tests for the activations module.
"""
import os
import pytest
import numpy as np
from models import activations
from .util import get_random_network_inputs_targets

# Define list of activation functions to be tested
act_func_list = [
    activations.identity,
    activations.logistic,
    activations.relu,
    activations.gaussian,
    activations.cauchy
]

# Get name of output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs")

@pytest.mark.parametrize("seed", [2144, 6646, 9914])
@pytest.mark.parametrize("act_func", act_func_list)
def test_shapes(seed, act_func):
    """
    Test that the output from each activation function and its derivatives are
    the correct shapes
    """
    _, x, _, _ = get_random_network_inputs_targets(seed)
    assert act_func(x).shape == x.shape
    assert act_func.dydx(x).shape == x.shape
    assert act_func.d2ydx2(x).shape == x.shape

@pytest.mark.parametrize("seed", [3995, 1218, 589])
@pytest.mark.parametrize("act_func", act_func_list)
def test_propagation(seed, act_func):
    """
    Test that the activation function can be used for forward and back
    propagation in a neural network model
    """
    n, x, t, _ = get_random_network_inputs_targets(seed, act_funcs=[act_func])
    n.forward_prop(x)
    n.back_prop(x, t)
    n.back_prop2(x, t)

@pytest.mark.parametrize("seed", [7279, 674, 4258])
@pytest.mark.parametrize("act_func", act_func_list)
def test_act_func_id(seed, act_func):
    """
    Test that the activation function IDs are self-consistent, IE that the
    correct activation function is returned from its id
    """
    act_func_id = act_func.get_id_from_func()
    act_func_from_id = activations.get_func_from_id(act_func_id)
    # Check that the types are consisitent
    assert type(act_func_from_id) is type(act_func)
    # Check that the outputs are consisitent
    _, x, _, _ = get_random_network_inputs_targets(seed)
    assert np.all(act_func_from_id(x) == act_func(x))

def test_act_func_ids_unique():
    """ Check that all of the activation function IDs are unique """
    id_list = [act_func.get_id_from_func() for act_func in act_func_list]
    assert len(set(id_list)) == len(id_list)

@pytest.mark.parametrize("act_func", act_func_list)
def test_plotting(act_func):
    """ Test plotting method for each activation function """
    act_func.plot(output_dir)
