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

# Initialise data, number of iterations, and results list
np.random.seed(9251)
sin_data = data.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)
n_iters = 10000
eval_every = n_iters // 20
results_list = []

for seed in [2295, 6997, 7681]:
    # Set the random seed
    np.random.seed(seed)
    # Generate random network and store initial parameters
    n = NeuralNetwork(
        1, 1, [10], [activations.Gaussian(), activations.Identity()]
    )
    w0 = n.get_parameter_vector().copy()
    # Call gradient descent function
    result_ls = optimisers.gradient_descent(
        n,
        sin_data,
        n_iters=n_iters,
        eval_every=eval_every,
        verbose=True,
        name="SGD with line search",
        line_search_flag=True
    )
    results_list.append(result_ls)
    # Try again without line search
    n.set_parameter_vector(w0)
    result_no_ls = optimisers.gradient_descent(
        n,
        sin_data,
        n_iters=n_iters,
        eval_every=eval_every,
        verbose=True,
        name="SGD without line search",
        line_search_flag=False
    )
    results_list.append(result_no_ls)

# Compare training curves
plotting.plot_training_curves(
    results_list,
    "Comparing line-search vs no line-search",
    output_dir,
    e_lims=[0, 0.2]
)
