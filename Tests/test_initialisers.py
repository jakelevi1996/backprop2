import os
import pytest
import numpy as np
import models, data
from .util import get_random_network_inputs_targets, get_output_dir_name

output_dir = get_output_dir_name()
test_initialisers_output_dir = os.path.join(output_dir, "Test initialisers")
if not os.path.isdir(test_initialisers_output_dir):
    os.makedirs(test_initialisers_output_dir)
    
def _print_pre_activation_statistics(nn, output_fname):
    np.set_printoptions(precision=3, linewidth=1000, suppress=True)
    output_path = os.path.join(test_initialisers_output_dir, output_fname)
    with open(output_path, "w") as f:
        for i, layer in enumerate(nn.layers):
            print("Layer %i pre-activation mean:" % i, file=f)
            print(layer.pre_activation.mean(axis=1, keepdims=True), file=f)
            print("Layer %i pre-activation STD:" % i, file=f)
            print(layer.pre_activation.std(axis=1, keepdims=True), file=f)

@pytest.mark.parametrize("seed", [9928, 1175, 3399])
def test_ConstantPreActivationStatistics(seed):
    """
    ...

    TODO: test with 0, 1, 2 hidden layers
    """
    np.random.seed(seed)
    input_dim = np.random.randint(2, 10)
    output_dim = np.random.randint(2, 10)
    N_D = np.random.randint(100, 200)
    x_lo = np.random.uniform(-10, 0)
    x_hi = np.random.uniform(0, 10)
    sin_data = data.Sinusoidal(input_dim, output_dim, N_D, 0, x_lo, x_hi)
    initialiser = models.initialisers.ConstantPreActivationStatistics(
        sin_data.x_train,
        sin_data.y_train
    )
    num_hidden_layers = np.random.randint(3, 6)
    num_hidden_units = np.random.randint(3, 6, num_hidden_layers)
    nn = models.NeuralNetwork(
        input_dim,
        output_dim,
        num_hidden_units,
        initialiser=initialiser
    )

    assert nn(sin_data.x_train).shape == sin_data.y_train.shape
    output_fname = "test_ConstantPreActivationStatistics, seed=%i.txt" % seed
    _print_pre_activation_statistics(nn, output_fname)
