import os
import numpy as np
import pytest
import optimisers
import models
from models import NeuralNetwork, activations
import data
import plotting
from .util import get_random_network

# Get name of output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs")

@pytest.mark.parametrize("seed", [5653, 9869, 2702])
def test_gradient_descent(seed):
    """
    Test gradient descent (no line-search is used, so there is no guarantee that
    each iteration reduces the error function).
    """
    # Set the random seed
    np.random.seed(seed)
    # Generate random number of iterations, network, data, and results file
    n_iters = np.random.randint(10, 20)
    n = get_random_network(input_dim=1, output_dim=1)
    sin_data = data.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)
    results_filename = "Test gradient descent without line-search.txt"
    results_path = os.path.join(output_dir, results_filename)
    results_file = open(results_path, "w")
    result = optimisers.Result(
        name="SGD without line search", 
        verbose=True,
        file=results_file
    )
    # Call gradient descent function
    result_ls = optimisers.gradient_descent(
        n,
        sin_data,
        terminator=optimisers.Terminator(i_lim=n_iters),
        evaluator=optimisers.Evaluator(i_interval=1),
        result=result
    )
    # Make sure each iteration reduces the training error
    for i in range(len(result_ls.train_errors) - 1):
        assert result_ls.train_errors[i + 1] < result_ls.train_errors[i]
    
    results_file.close()

@pytest.mark.parametrize("seed", [183, 3275, 9643])
@pytest.mark.parametrize("reuse_block_inds", [True, False])
def test_pbgn(seed, reuse_block_inds):
    """
    Test the Generalised Newton's method for optimisation, using parallel
    block-diagonal approximations.

    TODO: combine this and the gradient descent test (and any optimisers
    implemented in future, EG adam, PSO) in to a single parametrised test
    """
    # Set the random seed
    np.random.seed(seed)
    # Generate random number of iterations, network, data, and results file
    n_iters = np.random.randint(10, 20)
    n = NeuralNetwork(
        input_dim=1,
        output_dim=1,
        num_hidden_units=[4, 8, 6],
        act_funcs=[activations.gaussian, activations.identity]
    )
    sin_data = data.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)
    name = "Test PBGN without line-search, reuse_block_inds={}".format(
        reuse_block_inds
    )
    results_filename = "{}.txt".format(name)
    results_path = os.path.join(output_dir, results_filename)
    results_file = open(results_path, "w")
    result = optimisers.Result(
        name=name, 
        verbose=True,
        file=results_file
    )
    # Call gradient descent function
    result_ls = optimisers.generalised_newton(
        n,
        sin_data,
        terminator=optimisers.Terminator(i_lim=n_iters),
        evaluator=optimisers.Evaluator(i_interval=1),
        result=result,
        reuse_block_inds=reuse_block_inds
    )
    
    results_file.close()
    