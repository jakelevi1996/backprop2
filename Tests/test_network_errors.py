"""
Module to test errors that can be raised by the NeuralNetwork class. TODO:
update docstrings for methods in NeuralNetwork class detailing the errors that
are raised
"""

import numpy as np
import pytest
import models
from models import NeuralNetwork
from .util import get_random_network_inputs_targets, get_random_network, \
    get_random_inputs, get_random_targets, iterate_random_seeds

@iterate_random_seeds(3998, 3146, 9386)
def test_set_vector_errors():
    """
    Test the errors raised by the NeuralNetwork.set_parameter_vector method when
    there is an invalid argument
    """
    n = get_random_network()
    num_params = n.get_parameter_vector().size

    with pytest.raises(ValueError):
        # Wrong number of parameters
        n.set_parameter_vector(np.zeros(num_params - 3))
    
    with pytest.raises(ValueError):
        # Parameter array has the wrong shape (should be 1-dimensional)
        n.set_parameter_vector(np.zeros([1, num_params]))

    with pytest.raises(AttributeError):
        # Parameter vector is not a numpy array, so has no reshape method
        n.set_parameter_vector([0] * num_params)

@iterate_random_seeds(9045, 8503, 1546)
def test_invalid_act_function():
    """
    Test the errors raised by initialising an instance of the NeuralNetwork
    class when there is an invalid value for the act_funcs argument
    """
    with pytest.raises(TypeError):
        # act_funcs argument should be a list of activation function objects
        n = NeuralNetwork(act_funcs=models.activations.gaussian)
    
    n = NeuralNetwork(act_funcs=[models.activations._Gaussian])
    x, _ = get_random_inputs(n.input_dim)
    with pytest.raises(TypeError):
        # Network is initialised with the Gaussian class, not an instance
        n.forward_prop(x)

    n = NeuralNetwork(act_funcs=[None])
    x, _ = get_random_inputs(n.input_dim)
    with pytest.raises(TypeError):
        # Activation function None is not callable
        n.forward_prop(x)
        
    n = NeuralNetwork(act_funcs=[abs])
    x, N_D = get_random_inputs(n.input_dim)
    t = get_random_targets(n.output_dim, N_D)
    # Activation function abs is callable, so forward_prop is okay
    n.forward_prop(x)
    with pytest.raises(AttributeError):
        # Activation function abs has no dydx method, so backprop fails
        n.back_prop(x, t)

@iterate_random_seeds(8137, 9241, 8754)
def test_invalid_error_function():
    """
    Test the errors raised by initialising an instance of the NeuralNetwork
    class when there is an invalid value for the error_func argument
    """
    n = NeuralNetwork(error_func=models.errors._SumOfSquares)
    x, N_D = get_random_inputs(n.input_dim)
    t = get_random_targets(n.output_dim, N_D)
    with pytest.raises(TypeError):
        # Network is initialised with the SumOfSquares class, not an instance
        n.mean_error(t, x)

    n = NeuralNetwork(error_func=sum)
    x, N_D = get_random_inputs(n.input_dim)
    t = get_random_targets(n.output_dim, N_D)
    # Error function sum is callable, so mean_error is okay
    n.mean_error(t, x)
    with pytest.raises(AttributeError):
        # Error function sum has no dydx method, so backprop fails
        n.back_prop(x, t)

@iterate_random_seeds(6808, 8234, 4376)
def test_print_weights_grads_bad_file():
    """
    Test calling print_weights and print_grads methods with invalid file
    arguments
    """
    n = get_random_network()
    with pytest.raises(AttributeError):
        # file argument should be a file object returned by the open function
        n.print_weights(file="filename")
    with pytest.raises(AttributeError):
        # file argument should be a file object returned by the open function
        n.print_grads(file="filename")

@iterate_random_seeds(6216, 6733, 8485)
def test_print_grads_before_backprop_error():
    """
    Test that using the print_grads method before backpropagation raises an
    attribute error
    """
    n = get_random_network()
    with pytest.raises(AttributeError):
        n.print_grads()

@iterate_random_seeds(6926, 1825, 4717)
def test_forward_prop_bad_input_shape():
    pass

@iterate_random_seeds(2171, 7087, 3292)
def test_back_prop_bad_input_shape():
    pass

@iterate_random_seeds(8078, 3421, 2461)
def test_back_prop_bad_target_shape():
    pass
