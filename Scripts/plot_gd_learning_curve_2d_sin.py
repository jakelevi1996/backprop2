"""
Script to plot the learning curves for gradient descent with line-search on
sinusoidal data with 2 dimensional inputs, and 3 dimensional outputs.
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
output_dim = 3
sin_data = data.SinusoidalDataSet2DnD(
    nx0=50,
    x0lim=[-2, 2],
    nx1=50,
    x1lim=[-2, 2],
    noise_std=0.1,
    train_ratio=0.8,
    output_dim=output_dim
)
t_lim = 10
t_interval = t_lim / 50
results_list = []

for seed in [2295, 6997, 7681]:
    # Set the random seed
    np.random.seed(seed)
    # Generate random network and store initial parameters
    n = NeuralNetwork(
        input_dim=2,
        output_dim=output_dim,
        num_hidden_units=[20, 20],
        act_funcs=[activations.Cauchy(), activations.Identity()]
    )
    # Call gradient descent function
    result = optimisers.gradient_descent(
        n,
        sin_data,
        terminator=optimisers.Terminator(t_lim=t_lim),
        evaluator=optimisers.Evaluator(t_interval=t_interval),
        result=optimisers.Result(name="SGD with line search", verbose=True),
        line_search=optimisers.LineSearch()
    )
    results_list.append(result)

# Get name of output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs")

# Compare training curves
plotting.plot_training_curves(
    results_list,
    "Training curve for gradient descent on 2D sinusoidal data",
    output_dir,
    e_lims=[0, 4]
)
