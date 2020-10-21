"""
Script to compare learning curves for different initialisers, when training a
neural network on sinusoidal data with 2 dimensional inputs and 3 dimensional
outputs, using gradient descent with a line-search.

This script runs in approximately 30.730 s.

TODO: argparse wrapper
"""
import os
import numpy as np
from time import perf_counter
if __name__ == "__main__":
    import __init__
import models, data, optimisers, plotting

t_0 = perf_counter()

# Perform warmup experiment so process acquires priority
optimisers.warmup()

# Initialise data, time limit, and results list
np.random.seed(9251)
input_dim = 2
output_dim = 3
sin_data = data.Sinusoidal(
    input_dim=input_dim,
    output_dim=output_dim,
    n_train=2500
)
t_lim = 5
t_interval = t_lim / 50
results_list = []

for seed in [6666, 5990, 4910]:
    # Set the random seed
    np.random.seed(seed)
    # Generate random network
    n = models.NeuralNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        num_hidden_units=[20, 20],
        act_funcs=[models.activations.cauchy, models.activations.identity],
        initialiser=models.initialisers.ConstantPreActivationStatistics(
            sin_data.x_train,
            sin_data.y_train
        )
    )
    # Set name for experiment
    name = "Constant pre-activation statistics"
    # Call gradient descent function
    result = optimisers.gradient_descent(
        n,
        sin_data,
        terminator=optimisers.Terminator(t_lim=t_lim),
        evaluator=optimisers.Evaluator(t_interval=t_interval),
        result=optimisers.Result(name=name, verbose=True),
        line_search=optimisers.LineSearch(),
        batch_getter=optimisers.batch.ConstantBatchSize(50)
    )
    results_list.append(result)
    
    # Generate random network
    n = models.NeuralNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        num_hidden_units=[20, 20],
        act_funcs=[models.activations.cauchy, models.activations.identity],
        initialiser=models.initialisers.ConstantParameterStatistics()
    )
    # Set name for experiment
    name = "Constant parameter statistics"
    # Call gradient descent function
    result = optimisers.gradient_descent(
        n,
        sin_data,
        terminator=optimisers.Terminator(t_lim=t_lim),
        evaluator=optimisers.Evaluator(t_interval=t_interval),
        result=optimisers.Result(name=name, verbose=True),
        line_search=optimisers.LineSearch(),
        batch_getter=optimisers.batch.ConstantBatchSize(50)
    )
    results_list.append(result)
    
# Get name of output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs")

# Compare training curves
plotting.plot_training_curves(
    results_list,
    "Comparing initialisers for gradient descent on 2D sinusoidal data",
    output_dir,
    # e_lims=[0, 1.5],
    tp=0.5
)

print("Script run in {:.3f} s".format(perf_counter() - t_0))
