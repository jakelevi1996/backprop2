import numpy as np
from models import NeuralNetwork
import activations as a
import errors as e

def get_random_network(
    low=3,
    high=6,
    act_funcs=None,
    error_func=None
):
    """
    Generate a neural network with a random number of inputs, outputs, hidden
    layers, and number of units in each hidden layer
    """
    input_dim = np.random.randint(low, high)
    output_dim = np.random.randint(low, high)
    num_hidden_layers = np.random.randint(low, high)
    num_hidden_units = np.random.randint(low, high, num_hidden_layers)
    n = NeuralNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        num_hidden_units=num_hidden_units,
        act_funcs=act_funcs,
        error_func=error_func
    )
    return n

def get_random_inputs(input_dim, N_D_low=5, N_D_high=15):
    """ Generate random input data, with a random number of data points """
    N_D = np.random.randint(N_D_low, N_D_high)
    x = np.random.normal(size=[input_dim, N_D])
    return x, N_D

def get_random_targets(output_dim, N_D):
    """ Generate random target data """
    t = np.random.normal(size=[output_dim, N_D])
    return t

def get_random_network_inputs_targets(
    low=3,
    high=6,
    N_D_low=5,
    N_D_high=15,
    act_funcs=None,
    error_func=None
):
    """
    Wrapper for the following functions: get_random_network, get_random_inputs,
    get_random_targets
    """
    n = get_random_network(low, high, act_funcs, error_func)
    x, N_D = get_random_inputs(n.input_dim, N_D_low, N_D_high)
    t = get_random_targets(n.output_dim, N_D)
    return n, x, t, N_D

# TODO: write a decorator to repeat test functions with multiple random seeds
