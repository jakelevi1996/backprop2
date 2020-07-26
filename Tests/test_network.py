from os import makedirs
from os.path import dirname, abspath, join, isdir
import pytest
import numpy as np
from models import NeuralNetwork
import activations as a
import errors as e
from .util import get_random_network

# Get name of output directory, and create it if it doesn't already exist
current_dir = dirname(abspath(__file__))
output_dir = join(current_dir, "Outputs")
if not isdir(output_dir): makedirs(output_dir)

# Set numpy random seed and printing options
np.random.seed(1812)
np.set_printoptions(
    precision=3, linewidth=10000, suppress=True, threshold=10000
)

def test_network_init():
    """
    Test initialisation of a neural network, including multiple layers and
    different activation functions
    """
    # Set network parameters
    input_dim = 3
    output_dim = 2
    num_hidden_units = [4, 3, 2]
    act_funcs = [a.Relu(), a.Gaussian(), a.Logistic(), a.Identity()]
    error_func = e.SumOfSquares()
    weight_std = 2.3
    bias_std = 3.4
    
    # Initialise network
    n = NeuralNetwork(
        input_dim, output_dim, num_hidden_units, act_funcs, error_func,
        weight_std, bias_std
    )

def test_forward_propagation():
    """
    Test forward propagation, with multi-dimensional inputs and outputs,
    multiple hidden layers, and multiple data points, and assert that the output
    is the correct shape

    TODO: parameterise with different random seeds for different network shapes
    and sizes
    """
    n = get_random_network()
    N_D = 10
    x = np.random.normal(size=[n.input_dim, N_D])
    y = n.forward_prop(x)
    assert y.shape == (n.output_dim, N_D)

def test_back_propagation():
    """
    Test back propagation, with multi-dimensional inputs and outputs, and
    multiple data points
    """
    n = get_random_network()
    N_D = 10
    x = np.random.normal(size=[n.input_dim, N_D])
    t = np.random.normal(size=[n.output_dim, N_D])
    n.back_prop(x, t)

def test_backprop2():
    # TODO
    pass

def test_get_parameter_vector():
    """ Test the public method for getting the gradient vector """
    n = get_random_network()
    param_vector = n.get_parameter_vector()
    num_params = param_vector.size
    assert param_vector.shape == (num_params, )

def test_get_gradient_vector():
    """
    Test the public method for getting the parameter vector, and check that the
    gradient vector is approximately accurate using 1st order numerical
    differentiation
    """
    # Initialise network, inputs and targets
    n = get_random_network()
    N_D = 12
    x = np.random.normal(size=[n.input_dim, N_D])
    t = np.random.normal(size=[n.output_dim, N_D])
    # Get the gradient vector and check that it has the right shape
    gradient_vector = n.get_gradient_vector(x, t)
    num_params = gradient_vector.size
    assert gradient_vector.shape == (num_params, )

    # Change the weights and calculate the change in error
    dw_max = 1e-5
    tol = 1e-9
    E = n.mean_error(t, x)
    w = n.get_parameter_vector()
    dw = np.random.uniform(-dw_max, dw_max, gradient_vector.shape)
    n.set_parameter_vector(w + dw)
    E_new = n.mean_error(t, x)
    dE = E_new - E
    # Check that the gradient is consistent with numerical approximation
    assert abs(dE - np.dot(gradient_vector, dw)) < tol
    # Check that the answer isn't completely distorted by precision
    assert abs(dE) > max(abs(gradient_vector * dw))

def test_get_hessian():
    # TODO
    pass

def test_set_parameter_vector():
    """
    Test the public method for setting the parameter vector, and assert that the
    network outputs are different when the parameters are changed
    """
    # Initialise network and input
    n = get_random_network()
    N_D = 10
    x = np.random.normal(size=[n.input_dim, N_D])
    # Get old network params and output
    y_old = n(x)
    w_old = n.get_parameter_vector()
    # Set new network params
    w_new = w_old + 0.1
    n.set_parameter_vector(w_new)
    # Get new network output, and verify it is different to the old output
    y_new = n(x)
    assert not (y_old == y_new).all()

def test_mean_error():
    """
    Test calculating the mean error of a network with multi-dimensional inputs
    and outputs, and multiple data points
    """
    n = get_random_network()
    N_D = 10
    x = np.random.normal(size=[n.input_dim, N_D])
    t = np.random.normal(size=[n.output_dim, N_D])
    mean_error = n.mean_error(t, x)
    assert mean_error.shape == ()
    assert mean_error.size == 1

def test_save_load():
    # TODO
    pass

def test_print_weights():
    """
    Test printing the weights of a neural network with multi-dimensional inputs
    and outputs, and multiple layers, both to stdout and to a text file
    """
    n = get_random_network()
    # Print weights to stdout
    n.print_weights()
    # Print weights to file
    with open(join(output_dir, "weights.txt"), "w") as f:
        n.print_weights(f)
    
def test_print_grads():
    """
    Test printing the gradients of a neural network with multi-dimensional
    inputs and outputs, and multiple layers, both to stdout and to a text file
    """
    n = get_random_network()
    N_D = 10
    x = np.random.normal(size=[n.input_dim, N_D])
    t = np.random.normal(size=[n.output_dim, N_D])
    n.back_prop(x, t)
    # Print gradients to stdout
    n.print_grads()
    # Print gradients to file
    with open(join(output_dir, "gradients.txt"), "w") as f:
        n.print_grads(f)

def test_call_method():
    # TODO
    pass

def test_too_few_act_funcs():
    """
    Initialise a neural network with more hidden units than activation functions
    in the input arguments, and make sure that there are the correct number of
    layers, and that the activation function in each layer is set correctly
    """
    num_hidden_units = [3] * 10
    act_funcs = [a.Gaussian(), a.Identity()]
    n = NeuralNetwork(num_hidden_units=num_hidden_units, act_funcs=act_funcs)

    assert len(n.layers) == len(num_hidden_units) + 1
    
    for layer in n.layers[:-1]:
        assert type(layer.act_func) is a.Gaussian
    
    assert type(n.layers[-1].act_func) is a.Identity
    

def test_too_many_act_funcs():
    """
    Initialise a neural network with more activation functions than hidden units
    in the input arguments, and make sure that there are the correct number of
    layers, and that the activation function in each layer is set correctly
    """
    num_hidden_units = [3, 3]
    act_funcs = ([a.Gaussian()] * 10) + [a.Identity()]
    n = NeuralNetwork(num_hidden_units=num_hidden_units, act_funcs=act_funcs)

    assert len(n.layers) == len(num_hidden_units) + 1
    
    for layer in n.layers[:-1]:
        assert type(layer.act_func) is a.Gaussian
    
    assert type(n.layers[-1].act_func) is a.Identity
