"""
Module containing unit tests for the plotting module. The functions for plotting
activation and error functions are not tested here, but in the test modules for
the activation and error function modules. Similarly, the function for plotting
training curves is tested in the test module for the optimiser module.
"""
import os
import numpy as np
import pytest
import plotting, data
from optimisers import Result
from .util import get_random_network

# Get name of output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs")

@pytest.mark.parametrize("seed", [8974, 4798, 1812])
def test_plot_1D_regression(seed):
    """
    Test plotting function for data with 1 input dimension and 1 output
    dimension
    """
    np.random.seed(seed)
    # Generate training and test sets for 1D to 1D regression
    s11 = data.Sinusoidal(n_train=100, n_test=50, x_lo=0, x_hi=1)
    # Generate random predictions
    min_elem, max_elem = plotting.min_and_max(s11.x_train, s11.x_test)
    N_D = 200
    x_pred = np.linspace(min_elem, max_elem, N_D).reshape(1, -1)
    y_pred = np.random.normal(size=[1, N_D])
    # Call plotting function under test
    plotting.plot_1D_regression(
        "Random predictions 1D sinusoid",
        output_dir,
        s11,
        x_pred.ravel(),
        y_pred.ravel(),
        pred_marker="go"
    )


@pytest.mark.parametrize("seed, output_dim", [(1814, 1), (1743, 3)])
def test_plot_2D_nD_regression(seed, output_dim):
    """
    Test plotting function for data with 2 input dimensions and a variable
    number of output dimensions
    """
    np.random.seed(seed)
    input_dim = 2
    x_lo = -2
    x_hi = 2
    x_pred = np.linspace(x_lo, x_hi, 20)
    # Generate dataset and network
    s23 = data.Sinusoidal(
        input_dim=input_dim,
        output_dim=output_dim,
        n_train=200,
        x_lo=x_lo,
        x_hi=x_hi
    )
    model = get_random_network(
        input_dim=input_dim,
        output_dim=output_dim,
        low=2,
        high=3,
        weight_std=10
    )
    # Generate random predictions
    y_pred = np.random.normal(size=[output_dim, s23.x_test.shape[1]])
    # Call plotting function under test
    plotting.plot_2D_nD_regression(
        "Random predictions 2D-{}D sinusoid".format(output_dim),
        output_dir,
        n_output_dims=output_dim,
        dataset=s23,
        x_pred_0=x_pred,
        x_pred_1=x_pred,
        model=model,
    )

def test_plot_training_curves():
    np.random.seed(79)
    n_models = np.random.randint(2, 5)
    results_list = []

    for j in range(n_models):
        n_iters = np.random.randint(10, 20)
        output_dim = np.random.randint(2, 5)
        n = get_random_network(input_dim=2, output_dim=output_dim)
        d = data.Sinusoidal(input_dim=2, output_dim=output_dim, n_train=150)
        w = n.get_parameter_vector()
        result = Result(name="Network {}".format(j))
        result.begin()
        
        # Call the result.update method a few times
        for i in range(n_iters):
            n.set_parameter_vector(w + i)
            result.update(model=n, dataset=d, iteration=i)
        
        results_list.append(result)
    
    
    plotting.plot_training_curves(
        results_list,
        "Test plot_training_curves",
        output_dir,
        e_lims=None
    )
