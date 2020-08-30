import os
import numpy as np
if __name__ == "__main__":
    import __init__
from models import NeuralNetwork
import activations, data, optimisers, plotting

# Get name of output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs")

optimisers.warmup()

for seed in [7090, 2225]:
    # Set the random seed
    np.random.seed(seed)
    # Generate random network, data, and number of iterations
    n = NeuralNetwork(
        1, 1, [10], [activations.Gaussian(), activations.Identity()]
    )
    w0 = n.get_parameter_vector().copy()
    sin_data = data.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)
    n_iters = 500
    # Call gradient descent function
    result_ls = optimisers.gradient_descent(
        n,
        sin_data,
        n_iters=n_iters,
        eval_every=10,
        verbose=True,
        name="SGD with line search",
        line_search_flag=True
    )
    # Try again without line search
    n.set_parameter_vector(w0)
    result_no_ls = optimisers.gradient_descent(
        n,
        sin_data,
        n_iters=n_iters,
        eval_every=10,
        verbose=True,
        name="SGD without line search",
        line_search_flag=False
    )
    # Compare training curves
    plotting.plot_training_curves(
        [result_ls, result_no_ls],
        "Comparing line-search vs no line-search, seed = {}".format(seed),
        output_dir
    )
