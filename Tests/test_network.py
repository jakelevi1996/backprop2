from os import makedirs
from os.path import dirname, abspath, join, isdir
import pytest
import numpy as np
from models import NeuralNetwork
import activations as a
import errors as e
from .util import get_random_network_inputs_targets, iterate_random_seeds

# Get name of output directory, and create it if it doesn't already exist
current_dir = dirname(abspath(__file__))
output_dir = join(current_dir, "Outputs")
if not isdir(output_dir): makedirs(output_dir)

# Set numpy printing options
np.set_printoptions(
    precision=3, linewidth=10000, suppress=True, threshold=10000
)

@iterate_random_seeds(5920, 2788, 235)
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

@iterate_random_seeds(6588, 4626, 376)
def test_forward_propagation():
    """
    Test forward propagation, with multi-dimensional inputs and outputs,
    multiple hidden layers, and multiple data points, and assert that the output
    is the correct shape

    TODO: parameterise with different random seeds for different network shapes
    and sizes
    """
    n, x, _, N_D = get_random_network_inputs_targets()
    y = n.forward_prop(x)
    assert y.shape == (n.output_dim, N_D)

@iterate_random_seeds(8262, 6319, 3490)
def test_back_propagation():
    """
    Test back propagation, with multi-dimensional inputs and outputs, and
    multiple data points
    """
    n, x, t, _ = get_random_network_inputs_targets()
    n.back_prop(x, t)

@iterate_random_seeds(2770, 9098, 8645)
def test_backprop2():
    # TODO
    pass

@iterate_random_seeds(3052, 4449, 5555)
def test_get_parameter_vector():
    """ Test the public method for getting the gradient vector """
    n, _, _, _ = get_random_network_inputs_targets()
    param_vector = n.get_parameter_vector()
    num_params = param_vector.size
    assert param_vector.shape == (num_params, )

@iterate_random_seeds(6210, 9010, 9042)
def test_get_gradient_vector():
    """
    Test the public method for getting the parameter vector, and check that the
    gradient vector is approximately accurate using 1st order numerical
    differentiation
    """
    # Initialise network, inputs and targets
    n, x, t, _ = get_random_network_inputs_targets()
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

@iterate_random_seeds(5792, 1560, 3658)
def test_get_hessian():
    # TODO
    pass

@iterate_random_seeds(6563, 5385, 4070)
def test_set_parameter_vector():
    """
    Test the public method for setting the parameter vector, and assert that the
    network outputs are different when the parameters are changed
    """
    # Initialise network and input
    n, x, _, N_D = get_random_network_inputs_targets()
    # Get old network params and output
    y_old = n(x)
    w_old = n.get_parameter_vector()
    # Set new network params
    w_new = w_old + 0.1
    n.set_parameter_vector(w_new)
    # Get new network output, and verify it is different to the old output
    y_new = n(x)
    assert not (y_old == y_new).all()

@iterate_random_seeds(6544, 6633, 54)
def test_mean_error():
    """
    Test calculating the mean error of a network with multi-dimensional inputs
    and outputs, and multiple data points
    """
    n, x, t, _ = get_random_network_inputs_targets()
    mean_error = n.mean_error(t, x)
    assert mean_error.shape == ()
    assert mean_error.size == 1

@iterate_random_seeds(8585, 4350, 4503)
def test_save_load():
    # TODO
    pass

@iterate_random_seeds(2688, 3786, 6105)
def test_print_weights():
    """
    Test printing the weights of a neural network with multi-dimensional inputs
    and outputs, and multiple layers, both to stdout and to a text file
    """
    n, _, _, _ = get_random_network_inputs_targets()
    # Print weights to stdout
    n.print_weights()
    # Print weights to file
    with open(join(output_dir, "weights.txt"), "w") as f:
        n.print_weights(f)

@iterate_random_seeds(7629, 8258, 4020)
def test_print_grads():
    """
    Test printing the gradients of a neural network with multi-dimensional
    inputs and outputs, and multiple layers, both to stdout and to a text file
    """
    n, x, t, _ = get_random_network_inputs_targets()
    n.back_prop(x, t)
    # Print gradients to stdout
    n.print_grads()
    # Print gradients to file
    with open(join(output_dir, "gradients.txt"), "w") as f:
        n.print_grads(f)

@iterate_random_seeds(8658, 4807, 2199)
def test_call_method():
    # TODO
    pass

@iterate_random_seeds(8796, 238, 4789)
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
    
@iterate_random_seeds(6902, 8504, 1303)
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
