import os
import numpy as np
import pytest
import optimisers
from models import NeuralNetwork
import activations
import data
import plotting
from .util import get_random_network

# Get name of output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs")

@pytest.mark.parametrize("seed", [3217, 3132, 3523])
def test_gradient_descent_line_search(seed):
    """
    Test gradient descent, using a line-search. A line-search should guarantee
    that each iteration reduces the error function, so this is tested as
    asserted after calling the gradient_descent function.
    """
    # Set the random seed
    np.random.seed(seed)
    # Generate random number of iterations, network, data, and results file
    n_iters = np.random.randint(10, 20)
    n = NeuralNetwork(
        1, 1, [10], [activations.Gaussian(), activations.Identity()]
    )
    sin_data = data.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)
    results_filename = "Test gradient descent with line-search.txt"
    results_path = os.path.join(output_dir, results_filename)
    results_file = open(results_path, "w")
    # Call gradient descent function
    result_ls = optimisers.gradient_descent(
        n,
        sin_data,
        n_iters=n_iters,
        eval_every=1,
        verbose=True,
        name="SGD with line search",
        line_search_flag=True,
        result_file=results_file
    )
    # Make sure each iteration reduces the training error
    for i in range(len(result_ls.train_errors) - 1):
        assert result_ls.train_errors[i + 1] < result_ls.train_errors[i]
    
    results_file.close()

@pytest.mark.parametrize("seed", [5653, 9869, 2702])
def test_gradient_descent_no_line_search(seed):
    """
    Test gradient descent, without using a line-search, so there is no guarantee
    that each iteration reduces the error function.
    """
    # Set the random seed
    np.random.seed(seed)
    # Generate random number of iterations, network, data, and results file
    n_iters = np.random.randint(10, 20)
    n = NeuralNetwork(
        1, 1, [10], [activations.Gaussian(), activations.Identity()]
    )
    sin_data = data.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)
    results_filename = "Test gradient descent without line-search.txt"
    results_path = os.path.join(output_dir, results_filename)
    results_file = open(results_path, "w")
    # Call gradient descent function
    result_ls = optimisers.gradient_descent(
        n,
        sin_data,
        n_iters=n_iters,
        eval_every=1,
        verbose=True,
        name="SGD without line search",
        line_search_flag=False,
        result_file=results_file
    )
    # Make sure each iteration reduces the training error
    for i in range(len(result_ls.train_errors) - 1):
        assert result_ls.train_errors[i + 1] < result_ls.train_errors[i]
    
    results_file.close()

@pytest.mark.parametrize("seed", [183, 3275, 9643])
def test_pbgn_line_search(seed):
    """
    Test the Generalised Newton's method for optimisation, using parallel
    block-diagonal approximations and a line-search. A line-search should
    guarantee that each iteration reduces the error function, so this is tested
    as asserted after calling the gradient_descent function.

    TODO: combine this with the gradient descent with line-search test (and the
    no line search tests?) in to a single parametrised test
    """
    # Set the random seed
    np.random.seed(seed)
    # Generate random number of iterations, network, data, and results file
    n_iters = np.random.randint(10, 20)
    n = NeuralNetwork(
        1, 1, [4, 8, 6], [activations.Gaussian(), activations.Identity()]
    )
    sin_data = data.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)
    results_filename = "Test PBGN with line-search.txt"
    results_path = os.path.join(output_dir, results_filename)
    results_file = open(results_path, "w")
    # Call gradient descent function
    result_ls = optimisers.generalised_newton(
        n,
        sin_data,
        n_iters=n_iters,
        eval_every=1,
        verbose=True,
        name="SGD with line search",
        line_search_flag=True,
        result_file=results_file
    )
    # Make sure each iteration reduces the training error
    for i in range(len(result_ls.train_errors) - 1):
        assert result_ls.train_errors[i + 1] < result_ls.train_errors[i]
    
    results_file.close()
    