""" Module for containing general utilities for unit testing, EG generating
random NeuralNetwork models and data (both from existing Dataset classes and
also completely at random) """

import os
import hashlib
import numpy as np
import models, data

def set_random_seed_from_args(*args):
    """ Given a variable number of arguments, set an almost-surely unique
    random seed by using a tuple of the input arguments as the input to a hash
    function (and applying a 32-bit mask to the output from the hash function,
    because the input to the np.random.seed function "must be between 0 and
    2**32 - 1"). The inputs to this function could be for example a string
    representation of the test function, followed by the values of the
    arguments with which that test function is parametrised. """
    h = hashlib.sha256()
    for a in args:
        h.update(str(a).encode())
    b = h.digest()
    i = int.from_bytes(b, "big")
    seed = i & ((1 << 32) - 1)
    np.random.seed(seed)

def get_output_dir(subdir_name):
    """ Get name of output directory for a given test module, and create it if
    it doesn't exist """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "Outputs", subdir_name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    return output_dir

def get_random_network(
    low=3,
    high=6,
    act_funcs=None,
    error_func=None,
    input_dim=None,
    output_dim=None,
    initialiser=None,
):
    """ Generate a neural network with a random number of inputs, outputs,
    hidden layers, and number of units in each hidden layer """
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
        initialiser=initialiser,
    )
    return n

def get_random_inputs(input_dim, N_D_low=5, N_D_high=15):
    """ Generate random input data, with a random number of data points """
    N_D = np.random.randint(N_D_low, N_D_high)
    x = np.random.normal(size=[input_dim, N_D])
    return x, N_D

def get_random_targets(output_dim, N_D, error_func=None):
    """ Generate random target data. If an error function is specified, then
    the target data is generated to have the appropriate characteristics (EG 1D
    binary for binary classification, one-hot for multi-class classification)
    """
    if isinstance(error_func, data.BinaryClassification):
        assert output_dim == 1
        t = np.random.randint(2, size=[output_dim, N_D])
    elif isinstance(error_func, data.Classification):
        labels = np.random.randint(output_dim, size=N_D)
        t = np.zeros([output_dim, N_D])
        t[labels, np.arange(N_D)] = 1
    else:
        t = np.random.normal(size=[output_dim, N_D])
    
    return t

def get_random_network_inputs_targets(
    low=3,
    high=6,
    N_D_low=5,
    N_D_high=15,
    act_funcs=None,
    error_func=None,
    initialiser=None,
    input_dim=None,
    output_dim=None,
):
    """ Wrapper for the functions get_random_network, get_random_inputs, and
    get_random_targets. Return the outputs in a tuple """
    n = get_random_network(
        low=low,
        high=high,
        act_funcs=act_funcs,
        error_func=error_func,
        input_dim=input_dim,
        output_dim=output_dim,
        initialiser=initialiser,
    )
    x, N_D = get_random_inputs(n.input_dim, N_D_low, N_D_high)
    t = get_random_targets(n.output_dim, N_D)
    return n, x, t, N_D

def get_random_dataset_params(d_low=1, d_high=6, n_low=10, n_high=20):
    """ Randomly generate parameters needed to initialise a Dataset object,
    specifically the input and output dimensions, and the number of points in
    the training and test sets """
    input_dim   = np.random.randint(d_low, d_high)
    output_dim  = np.random.randint(d_low, d_high)
    n_train     = np.random.randint(n_low, n_high)
    n_test      = np.random.randint(n_low, n_high)
    return input_dim, output_dim, n_train, n_test

def get_dataset_and_name_from_type(dataset_type):
    """ Given the type (IE class) of a dataset, generate a random number of
    input and output dimensions and number of points in the training and test
    sets, initialise the dataset with those parameters, and return the
    initialised dataset """
    input_dim, output_dim, n_train, n_test = get_random_dataset_params()
    dataset_kwargs = {
        "input_dim":    input_dim,
        "n_train":      n_train,
        "n_test":       n_test,
    }
    if not issubclass(dataset_type, data.BinaryClassification):
        dataset_kwargs["output_dim"] = output_dim
    dataset = dataset_type(**dataset_kwargs)
    return dataset

def get_random_dataset():
    """ Return a dataset of a random type, with initialised with random
    parameters """
    dataset_type = np.random.choice(list(data.dataset_class_dict.values()))
    dataset = get_dataset_and_name_from_type(dataset_type)
    return dataset
