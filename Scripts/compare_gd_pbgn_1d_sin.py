"""
Script to compare learning curves for gradient descent vs parallel-block
generalised Newton's method, both with and without line-search, on 1D sinusoidal
data.
"""
import os
import numpy as np
if __name__ == "__main__":
    import __init__
from models import NeuralNetwork
import activations, data, optimisers, plotting

# Perform warmup experiment so process acquires priority
optimisers.warmup()

# Initialise data, time limit, and results list
np.random.seed(9251)
sin_data = data.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)
t_lim = 5
t_interval = t_lim / 50
results_list = []

for seed in [2295, 6997, 7681]:
    # Set the random seed
    np.random.seed(seed)
    # Generate random network and store initial parameters
    n = NeuralNetwork(
        input_dim=1,
        output_dim=1,
        num_hidden_units=[10],
        act_funcs=[activations.Gaussian(), activations.Identity()]
    )
    w0 = n.get_parameter_vector().copy()
    # Call gradient descent function
    result_gd_ls = optimisers.gradient_descent(
        n,
        sin_data,
        terminator=optimisers.Terminator(t_lim=t_lim),
        evaluator=optimisers.Evaluator(t_interval=t_interval),
        result=optimisers.Result(name="SGD with line search", verbose=True),
        line_search=optimisers.LineSearch()
    )
    results_list.append(result_gd_ls)
    # Try again without line search
    n.set_parameter_vector(w0)
    result_gd_no_ls = optimisers.gradient_descent(
        n,
        sin_data,
        terminator=optimisers.Terminator(t_lim=t_lim),
        evaluator=optimisers.Evaluator(t_interval=t_interval),
        result=optimisers.Result(name="SGD without line search", verbose=True),
        line_search=None
    )
    results_list.append(result_gd_no_ls)
    # Call generalised Newton function
    n.set_parameter_vector(w0)
    result_pbgn_ls = optimisers.generalised_newton(
        n,
        sin_data,
        terminator=optimisers.Terminator(t_lim=t_lim),
        evaluator=optimisers.Evaluator(t_interval=t_interval),
        result=optimisers.Result(name="PBGN with line search", verbose=True),
        line_search=optimisers.LineSearch()
    )
    results_list.append(result_pbgn_ls)
    # Try again without line search
    n.set_parameter_vector(w0)
    result_pbgn_no_ls = optimisers.generalised_newton(
        n,
        sin_data,
        terminator=optimisers.Terminator(t_lim=t_lim),
        evaluator=optimisers.Evaluator(t_interval=t_interval),
        result=optimisers.Result(name="PBGN without line search", verbose=True),
        line_search=None
    )
    results_list.append(result_pbgn_no_ls)

# Get name of output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs")

# Compare training curves
plotting.plot_training_curves(
    results_list,
    "Comparing gradient descent vs generalised Newton on 1D sinusoidal data",
    output_dir,
    e_lims=[0, 0.2]
)
