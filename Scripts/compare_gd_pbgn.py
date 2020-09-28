"""
Script to compare learning curves for gradient descent vs parallel-block
generalised Newton's method, both with and without line-search.
"""
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
    result_gd_ls = optimisers.gradient_descent(
        n,
        sin_data,
        terminator=optimisers.Terminator(t_lim=5),
        evaluator=optimisers.Evaluator(t_interval=5 / 50),
        verbose=True,
        name="SGD with line search",
        line_search_flag=True
    )
    results_list.append(result_gd_ls)
    # Try again without line search
    n.set_parameter_vector(w0)
    result_gd_no_ls = optimisers.gradient_descent(
        n,
        sin_data,
        terminator=optimisers.Terminator(t_lim=5),
        evaluator=optimisers.Evaluator(t_interval=5 / 50),
        verbose=True,
        name="SGD without line search",
        line_search_flag=False
    )
    results_list.append(result_gd_no_ls)
    # Call generalised Newton function
    n.set_parameter_vector(w0)
    result_pbgn_ls = optimisers.generalised_newton(
        n,
        sin_data,
        terminator=optimisers.Terminator(t_lim=5),
        evaluator=optimisers.Evaluator(t_interval=5 / 50),
        verbose=True,
        name="PBGN with line search",
        line_search_flag=True
    )
    results_list.append(result_pbgn_ls)
    # Try again without line search
    n.set_parameter_vector(w0)
    result_pbgn_no_ls = optimisers.generalised_newton(
        n,
        sin_data,
        terminator=optimisers.Terminator(t_lim=5),
        evaluator=optimisers.Evaluator(t_interval=5 / 50),
        verbose=True,
        name="PBGN without line search",
        line_search_flag=False
    )
    results_list.append(result_pbgn_no_ls)

# Compare training curves
plotting.plot_training_curves(
    results_list,
    "Comparing gradient descent vs generalised Newton",
    output_dir,
    e_lims=[0, 0.2]
)
