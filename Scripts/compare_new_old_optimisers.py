"""
Compare the new gradient descent optimiser function with the old gradient
descent optimiser function.

TODO: also compare new and old optimisers without line search
"""
import os
import numpy as np
if __name__ == "__main__":
    import __init__
from models import NeuralNetwork
import optimisers, optimisers_old, activations, data, plotting

# Get name of output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs")

# Initialise data and number of iterations
sin_data = data.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)
# n_iters = 500
n_iters = 10000
eval_every = n_iters // 20

# Perform warmup routine
optimisers.warmup()

for seed in [3530, 8866, 1692]:
    # Set the random seed
    np.random.seed(seed)
    # Generate random network, data, and number of iterations
    n = NeuralNetwork(
        1, 1, [10], [activations.Gaussian(), activations.Identity()]
    )
    w0 = n.get_parameter_vector().copy()
    # Call new gradient descent function
    result_new_optimiser = optimisers.gradient_descent(
        n,
        sin_data,
        n_iters=n_iters,
        eval_every=eval_every,
        verbose=True,
        name="New SGD function",
        line_search_flag=True
    )
    # Call old gradient descent function
    n.set_parameter_vector(w0)
    result_old_optimiser = optimisers_old.sgd_2way_tracking(
        n,
        sin_data,
        n_iters,
        eval_every,
        name="Old SGD function"
    )
    # Compare training curves
    plotting.plot_training_curves(
        [result_new_optimiser, result_old_optimiser],
        "Comparing new VS old optimisers, seed = {}".format(seed),
        output_dir,
        e_lims=[0, 0.02]
    )
