import os
import numpy as np
import pytest
import optimisers as o
import models as m
import activations as a
import data as d
import plotting
from .util import get_random_network

# Get name of output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs")

@pytest.mark.parametrize("seed", [7090, 2225])
def test_gradient_descent(seed):
    """
    TODO: only a few iterations here, with random network. For real comparison,
    use a script.
    """
    # Set the random seed
    np.random.seed(seed)
    # Generate random network, data, and number of iterations
    n = m.NeuralNetwork(1, 1, [10], [a.Gaussian(), a.Identity()])
    w0 = n.get_parameter_vector().copy()
    sin_data = d.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)
    n_iters = 500
    # Call gradient descent function
    result_ls = o.gradient_descent(
        n,
        sin_data,
        n_iters=n_iters,
        eval_every=10,
        verbose=True,
        name="New SGD with line search",
        line_search_flag=True
    )
    # Make sure each iteration reduces the training error
    for i in range(len(result_ls.train_errors) - 1):
        assert result_ls.train_errors[i + 1] < result_ls.train_errors[i]
    # Plot predictions
    x_pred = np.linspace(-2, 2).reshape(1, -1)
    y_pred = n.forward_prop(x_pred)
    plotting.plot_1D_regression(
        "Test gradient descent predictions",
        output_dir,
        sin_data,
        x_pred,
        y_pred
    )
    # Try again without line search
    n.set_parameter_vector(w0)
    result_no_ls = o.gradient_descent(
        n,
        sin_data,
        n_iters=n_iters,
        eval_every=10,
        verbose=True,
        name="New SGD without line search",
        line_search_flag=False
    )
    # Compare training curves
    plotting.plot_training_curves(
        [result_ls, result_no_ls],
        "Learning curves",
        output_dir
    )

def test_old_sgd_function():
    # TODO
    pass

# # warmup()
# np.random.seed(0)
# sin_data = d.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)
# n = m.NeuralNetwork(1, 1, [20])
# w = n.get_parameter_vector().copy()

# # stochastic_gradient_descent(n, sin_data, 100, 10)
# # n.set_parameter_vector(w)
# # sgd_2way_tracking(n, sin_data, 100, 10)

# o.stochastic_gradient_descent(n, sin_data, 10000, 1000)
# n.set_parameter_vector(w)
# o.gradient_descent(n, sin_data, n_iters=10000, eval_every=1000, verbose=True,
# name="New SGD")
# n.set_parameter_vector(w)
# o.sgd_2way_tracking(n, sin_data, 10000, 1000)
# n.set_parameter_vector(w)
# o.gradient_descent(n, sin_data, n_iters=10000, eval_every=1000, verbose=True,
# line_search_flag=True, name="New SGD + LS", learning_rate=1.0,
# alpha=0.8, beta=0.5, t_lim=10)
