"""
Find best parameters for gradient descent with line-search, by trying different
combinations and plotting the results. The different parameters that are
compared are:
-   Number of units
-   Number of layers
-   s0 (initial step size)
-   alpha (threshold for backtracking)
-   beta (ratio of changes in step size)
-   Activation function

Results are compared by plotting final performance after a fixed length of time
allowed for optimisation.

This script has a command-line interface. Below are some examples for calling
this script:

    python "Scripts\Compare parameters\gradient_descent.py"

    python "Scripts\Compare parameters\gradient_descent.py" -i2 -o3

    python "Scripts\Compare parameters\gradient_descent.py" -t"0.1"

    python "Scripts\Compare parameters\gradient_descent.py" -b

To get help information for the available arguments, use the following command:

    python "Scripts\Compare parameters\gradient_descent.py" -h
"""
import os
import argparse
import numpy as np
if __name__ == "__main__":
    import __init__
from models import NeuralNetwork
import models, data, optimisers
from run_all_experiments import run_all_experiments

def main(
    input_dim,
    output_dim,
    n_train,
    t_lim,
    t_eval,
    n_repeats,
    find_best_params
):
    # Get name of output directory
    param_str = (
        "input_dim = %i, output_dim = %i, n_train = %i, t_lim = %.2f, "
        "num_repeats = %i, find_best_params = %s" % (
            input_dim,
            output_dim,
            n_train,
            t_lim,
            n_repeats,
            find_best_params
        )
    )
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(
        current_dir,
        "Outputs",
        "Gradient descent",
        param_str
    )
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Initialise dictionary of parameter names, default values, values to test
    all_experiments_dict = {
        "num_units":    {"default": 10,    "range": [5, 10, 15, 20]},
        "num_layers":   {"default": 1,     "range": [1, 2, 3]},
        "log10_s0":     {"default": 0,     "range": np.linspace(-1, 3, 5)},
        "alpha":        {"default": 0.5,   "range": np.arange(0.5, 1, 0.1)},
        "beta":         {"default": 0.5,   "range": np.arange(0.5, 1, 0.1)},
        "max_steps":    {"default": 10,    "range": [5, 10, 15, 20]},
        "act_func":     {
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
    sin_data = data.Sinusoidal(
        input_dim,
        output_dim,
        n_train,
        x_lo=-2,
        x_hi=2,
        freq=1
    )

    # Define function to be run for each experiment
    def run_experiment(
        dataset,
        num_units,
        num_layers,
        log10_s0,
        alpha,
        beta,
        act_func,
        max_steps
    ):
        print(num_units, num_layers, log10_s0, alpha, beta, act_func)
        n = NeuralNetwork(
            input_dim=1,
            output_dim=1,
            num_hidden_units=[num_units for _ in range(num_layers)],
            act_funcs=[act_func, models.activations.identity]
        )
        result = optimisers.gradient_descent(
            n,
            dataset,
            terminator=optimisers.Terminator(t_lim=t_lim),
            evaluator=optimisers.Evaluator(t_interval=t_eval),
            line_search=optimisers.LineSearch(
                s0=pow(10, log10_s0), 
                alpha=alpha, 
                beta=beta,
                max_its=max_steps
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
        output_dir,
        n_repeats,
        find_best_parameters=find_best_params
    )

if __name__ == "__main__":
    # Define CLI using argparse
    parser = argparse.ArgumentParser(
        description="Compare parameters for gradient descent with a line search"
    )

    parser.add_argument(
        "-i",
        "--input_dim",
        help="Number of input dimensions for the data set",
        default=1,
        type=int
    )
    parser.add_argument(
        "-o",
        "--output_dim",
        help="Number of output dimensions for the data set",
        default=1,
        type=int
    )
    parser.add_argument(
        "-n",
        "--n_train",
        help="Number of data points in the training set",
        default=50,
        type=int
    )
    parser.add_argument(
        "-t",
        "--t_lim",
        help="How long to run each experiment for in seconds",
        default=.03,
        type=float
    )
    parser.add_argument(
        "-e",
        "--t_eval",
        help=(
            "How frequently to evaluate the performance of each model in "
            "seconds"
        ),
        default=0.5,
        type=float
    )
    parser.add_argument(
        "-r",
        "--n_repeats",
        help="Number of repeats to perform of each experiment",
        default=5,
        type=int
    )
    parser.add_argument(
        "-b",
        "--find_best_params",
        help="Number of repeats to perform of each experiment",
        action="store_true"
    )

    # Parse arguments
    args = parser.parse_args()

    # Call main function using command-line arguments
    main(
        args.input_dim,
        args.output_dim,
        args.n_train,
        args.t_lim,
        args.t_eval,
        args.n_repeats,
        args.find_best_params
    )
