import os
import pytest
import numpy as np
from models import NeuralNetwork
import activations as a
import errors as e
from .util import get_random_network_inputs_targets, iterate_random_seeds

# Get name of output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs")



@pytest.mark.parametrize("seed", [6588, 4626, 376])
def test_forward_propagation(seed):
    """
    Test forward propagation, with multi-dimensional inputs and outputs,
    multiple hidden layers, and multiple data points, and assert that the output
    is the correct shape
    """
    n, x, _, N_D = get_random_network_inputs_targets(seed)
    y = n.forward_prop(x)
    assert y.shape == (n.output_dim, N_D)

@pytest.mark.parametrize("seed", [8262, 6319, 3490])
def test_back_propagation(seed):
    """
    Test back propagation, with multi-dimensional inputs and outputs, and
    multiple data points, and assert that the appropriate attributes of each of
    the network layers have the correct shape
    """
    n, x, t, N_D = get_random_network_inputs_targets(seed)
    n.forward_prop(x)
    n.back_prop(x, t)
    # Iterate through each layer
    for layer in n.layers:
        # Assert that the gradients in each layer have the correct shape
        assert layer.delta.shape == (layer.output_dim, N_D)
        assert layer.w_grad.shape == (layer.output_dim, layer.input_dim, N_D)
        assert layer.b_grad.shape == (layer.output_dim, N_D)


@pytest.mark.parametrize("seed", [2770, 9098, 8645])
def test_backprop2(seed):
    """
    Test 2nd order back propagation, with multi-dimensional inputs and outputs,
    and multiple data points, and assert that the epsilons in each network
    layer have the correct shape
    """
    n, x, t, N_D = get_random_network_inputs_targets(seed)
    n.forward_prop(x)
    n.back_prop(x, t)
    n.back_prop2(x, t)
    # Iterate through each layer
    for layer in n.layers:
        # Assert that the gradients in each layer have the correct shape
        assert layer.epsilon.shape == (layer.output_dim, layer.output_dim, N_D)

@pytest.mark.parametrize("seed", [3052, 4449, 5555])
def test_get_parameter_vector(seed):
    """ Test the public method for getting the gradient vector """
    n, _, _, _ = get_random_network_inputs_targets(seed)
    param_vector = n.get_parameter_vector()
    num_params = param_vector.size
    assert param_vector.shape == (num_params, )

@pytest.mark.parametrize("seed", [6210, 9010, 9042])
def test_get_gradient_vector(seed):
    """
    Test the public method for getting the parameter vector, and check that the
    gradient vector is approximately accurate using 1st order numerical
    differentiation
    """
    # Initialise network, inputs and targets
    n, x, t, _ = get_random_network_inputs_targets(seed)
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

@pytest.mark.parametrize("seed", [5792, 1560, 3658])
def test_get_hessian_blocks(seed):
    """
    Check that get_hessian_blocks returns symmetric Hessian blocks with the
    correct shape, and that the corresponding list of inds for each block has
    the correct number of elements, and that each element is unique
    """

    # Initialise network, inputs and targets
    n, x, t, _ = get_random_network_inputs_targets(seed)
    
    # Set block inds for get_hessian_blocks method
    max_block_size = np.random.randint(3, 6)
    weight_inds_list = [
        np.array_split(
            np.random.permutation(layer.num_weights),
            np.ceil(layer.num_weights / max_block_size)
        ) for layer in n.layers
    ]
    bias_inds_list = [
        np.array_split(
            np.random.permutation(layer.num_bias),
            np.ceil(layer.num_bias / max_block_size)
        ) for layer in n.layers
    ]
    
    # Get Hessian blocks
    n.forward_prop(x)
    n.back_prop(x, t)
    hess_block_list, hess_inds_list = n.get_hessian_blocks(
        x,
        t,
        weight_inds_list,
        bias_inds_list
    )
    
    # Calculate expected shapes
    expected_shapes = []
    for layer_w_inds, layer_b_inds in zip(weight_inds_list, bias_inds_list):
        expected_shapes += [(block.size, block.size) for block in layer_w_inds]
        expected_shapes += [(block.size, block.size) for block in layer_b_inds]
    
    # Iterate through each block and expected shape
    for block, shape in zip(hess_block_list, expected_shapes):
        # Check the shape is as expected
        assert block.shape == shape
        # Check that the Hessian block is symmetric
        assert np.allclose(block, block.T)
    
    # Check that Hessian inds are all unique and the right length
    unpacked_hess_inds_list = [
        ind
        for block_inds_list in hess_inds_list
        for ind in block_inds_list
    ]
    assert len(unpacked_hess_inds_list) == len(set(unpacked_hess_inds_list))
    assert len(unpacked_hess_inds_list) == n.num_params

@pytest.mark.parametrize("seed", [6563, 5385, 4070])
def test_set_parameter_vector(seed):
    """
    Test the public method for setting the parameter vector, and assert that the
    network outputs are different when the parameters are changed
    """
    # Initialise network and input
    n, x, _, N_D = get_random_network_inputs_targets(seed)
    # Get a copy of the old network params and output
    y_old = n(x)
    w_old = n.get_parameter_vector().copy()
    # Set new network params
    w_new = w_old + 0.1
    n.set_parameter_vector(w_new)
    # Check that the new parameter vector is now in the network
    assert (n.get_parameter_vector() == w_new).all()
    assert (n.get_parameter_vector() != w_old).all()
    # Get new network output, and verify it is different to the old output
    y_new = n(x)
    assert not (y_old == y_new).all()

@pytest.mark.parametrize("seed", [6544, 6633, 54])
def test_mean_error(seed):
    """
    Test calculating the mean error of a network with multi-dimensional inputs
    and outputs, and multiple data points
    """
    n, x, t, _ = get_random_network_inputs_targets(seed)
    mean_error = n.mean_error(t, x)
    assert mean_error.shape == ()
    assert mean_error.size == 1

@pytest.mark.parametrize("seed", [8585, 4350, 4503])
def test_save_load(seed):
    # TODO
    pass

@pytest.mark.parametrize("seed", [2688, 3786, 6105])
def test_print_weights(seed):
    """
    Test printing the weights of a neural network with multi-dimensional inputs
    and outputs, and multiple layers, both to stdout and to a text file
    """
    n, _, _, _ = get_random_network_inputs_targets(seed)
    # Print weights to stdout
    n.print_weights()
    # Print weights to file
    with open(os.path.join(output_dir, "weights.txt"), "w") as f:
        n.print_weights(f)

@pytest.mark.parametrize("seed", [7629, 8258, 4020])
def test_print_grads(seed):
    """
    Test printing the gradients of a neural network with multi-dimensional
    inputs and outputs, and multiple layers, both to stdout and to a text file
    """
    n, x, t, _ = get_random_network_inputs_targets(seed)
    n.forward_prop(x)
    n.back_prop(x, t)
    # Print gradients to stdout
    n.print_grads()
    # Print gradients to file
    with open(os.path.join(output_dir, "gradients.txt"), "w") as f:
        n.print_grads(f)

@pytest.mark.parametrize("seed", [8658, 4807, 2199])
def test_call_method(seed):
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
