import os
import numpy as np
import pytest
import optimisers
from models import NeuralNetwork
import activations
import data
from .util import get_random_network

# Get name of output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs")

@pytest.mark.parametrize("seed", [3217, 3132, 3523])
def test_gradient_descent_line_search(seed):
    """
    Test gradient descent, using a line-search. A line-search should guarantee
    that each iteration reduces the error function, so this is tested using
    assert statements after calling the gradient_descent function.
    """
    # Set the random seed
    np.random.seed(seed)
    # Generate random number of iterations, network, data, and results file
    n_iters = np.random.randint(10, 20)
    n = get_random_network(input_dim=1, output_dim=1)
    sin_data = data.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)
    results_filename = "Test gradient descent with line-search.txt"
    results_path = os.path.join(output_dir, results_filename)
    results_file = open(results_path, "w")
    result = optimisers.Result(
        name="SGD with line search", 
        verbose=True,
        file=results_file
    )
    # Call gradient descent function
    result_ls = optimisers.gradient_descent(
        n,
        sin_data,
        terminator=optimisers.Terminator(i_lim=n_iters),
        evaluator=optimisers.Evaluator(i_interval=1),
        line_search=optimisers.LineSearch(),
        result=result
    )
    # Make sure each iteration reduces the training error
    for i in range(len(result_ls.train_errors) - 1):
        assert result_ls.train_errors[i + 1] < result_ls.train_errors[i]
    
    results_file.close()