"""
Module to test errors that can be raised by the NeuralNetwork class. TODO:
docstrings and comments, also update docstrings for methods in NeuralNetwork
class detailing the errors that are raised
"""

import numpy as np
import pytest
from models import NeuralNetwork
from .util import get_random_network

def test_set_vector_wrong_size():
    n = get_random_network()
    num_params = n.get_parameter_vector().size

    with pytest.raises(ValueError):
        n.set_parameter_vector(np.zeros(num_params - 3))
    
    with pytest.raises(ValueError):
        n.set_parameter_vector(np.zeros([1, num_params]))

    with pytest.raises(AttributeError):
        n.set_parameter_vector([0] * num_params)

def test_invalid_act_function():
    n = NeuralNetwork(act_funcs=[None])
    N_D = 10
    x = np.random.normal(size=[n.input_dim, N_D])
    t = np.random.normal(size=[n.output_dim, N_D])
    
    with pytest.raises(TypeError):
        n.forward_prop(x)
        
    n = NeuralNetwork(act_funcs=[sum])
    N_D = 10
    x = np.random.normal(size=[n.input_dim, N_D])
    t = np.random.normal(size=[n.output_dim, N_D])
    
    n.forward_prop(x)

    with pytest.raises(AttributeError):
        n.back_prop(x, t)

def test_invalid_error_function():
    n = NeuralNetwork(error_func=None)
    N_D = 10
    x = np.random.normal(size=[n.input_dim, N_D])
    t = np.random.normal(size=[n.output_dim, N_D])
    
    with pytest.raises(TypeError):
        n.mean_error(t, x)
        
    with pytest.raises(AttributeError):
        n.back_prop(x, t)

    n = NeuralNetwork(error_func=sum)
    N_D = 10
    x = np.random.normal(size=[n.input_dim, N_D])
    t = np.random.normal(size=[n.output_dim, N_D])
    
    n.mean_error(t, x)

    with pytest.raises(AttributeError):
        n.back_prop(x, t)

def test_print_weights_bad_file():
    n = get_random_network()
    with pytest.raises(AttributeError):
        n.print_weights(file="filename")

def test_print_grads_before_backprop_error():
    """
    Test that using the print_grads method before backpropagation raises an
    attribute error
    """
    n = get_random_network()
    with pytest.raises(AttributeError):
        n.print_grads()
