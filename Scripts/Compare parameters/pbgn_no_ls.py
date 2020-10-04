"""
Script to compare parameters for parallel-block generalised-Newton's method
without a line-search on 1D sinusoidal data.

This script runs in approximately 7 minutes and 3 seconds.
"""
import os
import numpy as np
if __name__ == "__main__":
    import __init__
from models import NeuralNetwork
import activations, data, optimisers
from run_all_experiments import run_all_experiments

# Get name of output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs", "PBGN (no line-search)")
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# Initialise dictionary of parameter names, default values, and values to test
all_experiments_dict = {
    "num_units":            {"default": 10,   "range": [5, 10, 15, 20]},
    "num_layers":           {"default": 1,    "range": [1, 2, 3]},
    "log10_learning_rate":  {"default": -1,   "range": np.linspace(-3, 1, 5)},
    "max_block_size":       {"default": 7,    "range": np.arange(5, 11, 1)},
    "log10_max_step":       {"default": 0,    "range": np.linspace(-1, 1, 5)},
    "reuse_block_inds":     {"default": True, "range": [True, False]},
    "act_func":             {
        "default": activations.Gaussian(),
        "range": [
            activations.Gaussian(),
            activations.Cauchy(),
            activations.Logistic(),
            activations.Relu(),
        ]
    },
}

# Initialise data set
np.random.seed(6763)
sin_data = data.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)

# Define function to be run for each experiment
def run_experiment(
    dataset,
    num_units,
    num_layers,
    log10_learning_rate,
    max_block_size,
    log10_max_step,
    reuse_block_inds,
    act_func
):
    n = NeuralNetwork(
        input_dim=1,
        output_dim=1,
        num_hidden_units=[num_units for _ in range(num_layers)],
        act_funcs=[act_func, activations.Identity()]
    )
    result = optimisers.generalised_newton(
        n,
        dataset,
        learning_rate=pow(10, log10_learning_rate),
        max_block_size=max_block_size,
        max_step=pow(10, log10_max_step),
        terminator=optimisers.Terminator(t_lim=3),
        evaluator=optimisers.Evaluator(t_interval=0.1),
        line_search=None
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
