"""
Find best parameters for gradient descent with line-search, by trying different
combinations and plotting the results. The different parameters that are
compared are:
-   number of units
-   number of layers
-   learning_rate
-   s0 (initial step size)
-   alpha (threshold for backtracking)
-   beta (ratio of changes in step size)
-   Activation function

Results are compared by plotting final performance after a fixed length of time
allowed for optimisation.

TODO:
-   Add argparse wrapper for this script, so experiments can be configured
    from the command line
"""
import os
import numpy as np
if __name__ == "__main__":
    import __init__
from models import NeuralNetwork
import models, data, optimisers
from run_all_experiments import run_all_experiments

# Get name of output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs", "Gradient descent")
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# Initialise dictionary of parameter names, default values, and values to test
all_experiments_dict = {
    "num_units":            {"default": 10,    "range": [5, 10, 15, 20]},
    "num_layers":           {"default": 1,     "range": [1, 2, 3]},
    "log10_learning_rate":  {"default": -1,    "range": np.linspace(-3, 1, 5)},
    "log10_s0":             {"default": 0,     "range": np.linspace(-1, 3, 5)},
    "alpha":                {"default": 0.5,   "range": np.arange(0.5, 1, 0.1)},
    "beta":                 {"default": 0.5,   "range": np.arange(0.5, 1, 0.1)},
    "act_func":             {
        "default": models.activations.gaussian,
        "range": [
            models.activations.gaussian,
            models.activations.cauchy,
            models.activations.logistic,
            models.activations.relu,
        ]
    },
}

# Initialise data set
np.random.seed(6763)
sin_data = data.Sinusoidal(x_lo=-2, x_hi=2, freq=1)

# Define function to be run for each experiment
def run_experiment(
    dataset,
    num_units,
    num_layers,
    log10_learning_rate,
    log10_s0,
    alpha,
    beta,
    act_func
):
    n = NeuralNetwork(
        input_dim=1,
        output_dim=1,
        num_hidden_units=[num_units for _ in range(num_layers)],
        act_funcs=[act_func, models.activations.identity]
    )
    result = optimisers.gradient_descent(
        n,
        dataset,
        learning_rate=pow(10, log10_learning_rate),
        terminator=optimisers.Terminator(t_lim=3),
        evaluator=optimisers.Evaluator(t_interval=0.5),
        line_search=optimisers.LineSearch(
            s0=pow(10, log10_s0), 
            alpha=alpha, 
            beta=beta
        )
    )
    return result

# Call warmup function
optimisers.warmup()

# Call function to run all experiments
run_all_experiments(
    all_experiments_dict,
    run_experiment,
    sin_data,
    output_dir
)
