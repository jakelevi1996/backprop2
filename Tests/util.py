"""
Module for containing general utilities for unit testing, EG generating random
NeuralNetwork models and data (both from existing Dataset classes and also
completely at random)
"""

import numpy as np
import models, data

def get_random_network(
    low=3,
    high=6,
    act_funcs=None,
    error_func=None,
    input_dim=None,
    output_dim=None,
    initialiser=None
):
    """
    Generate a neural network with a random number of inputs, outputs, hidden
    layers, and number of units in each hidden layer
    """
    if input_dim is None:
        input_dim = np.random.randint(low, high)
    if output_dim is None:
        output_dim = np.random.randint(low, high)
    num_hidden_layers = np.random.randint(low, high)
    num_hidden_units = np.random.randint(low, high, num_hidden_layers)
    n = models.NeuralNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        num_hidden_units=num_hidden_units,
        act_funcs=act_funcs,
        error_func=error_func,
        initialiser=initialiser
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
    seed,
    low=3,
    high=6,
    N_D_low=5,
    N_D_high=15,
    act_funcs=None,
    error_func=None,
    initialiser=None
):
    """
    Wrapper for the following functions: np.random.seed, get_random_network,
    get_random_inputs, get_random_targets. Return the outputs in a tuple
    """
    np.random.seed(seed)
    n = get_random_network(
        low,
        high,
        act_funcs,
        error_func,
        initialiser=initialiser
    )
    x, N_D = get_random_inputs(n.input_dim, N_D_low, N_D_high)
    t = get_random_targets(n.output_dim, N_D)
    return n, x, t, N_D

dataset_list = [
    "1x1_sinusoidal_set_freq",
    "1x1_sinusoidal_random_freq",
    "2x1_sinusoidal",
    "2x4_sinusoidal",
]

def get_dataset(dataset_str):
    if dataset_str == "1x1_sinusoidal_set_freq":
        return data.Sinusoidal(input_dim=1, output_dim=1, freq=1.1)
    elif dataset_str == "1x1_sinusoidal_random_freq":
        return data.Sinusoidal(input_dim=1, output_dim=1)
    elif dataset_str == "2x1_sinusoidal":
        return data.Sinusoidal(input_dim=2, output_dim=1)
    elif dataset_str == "2x4_sinusoidal":
        return data.Sinusoidal(input_dim=2, output_dim=4)
    else:
        raise ValueError("Invalid input string")
