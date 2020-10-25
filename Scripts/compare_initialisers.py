"""
Script to compare learning curves for different initialisers, when training a
neural network on sinusoidal data using gradient descent with a line-search.

This script runs in approximately 30.730 s.

Below are some examples for calling this script:

    python Scripts\compare_initialisers.py 2 3 2500 50 5 20,20 0 4 3

    python Scripts\compare_initialisers.py 2 3 2500 50 1 20,20 0 4 3

To get help information for the available arguments, use the following command:

    python Scripts\compare_initialisers.py -h

TODO: add command line arguments for initialiser statistics, format output
title, run more experiments with different parameters (varying EG initialiser
statistics, batch sizes,)
"""
import os
from argparse import ArgumentParser
from time import perf_counter
import numpy as np
if __name__ == "__main__":
    import __init__
import models, data, optimisers, plotting

def main(
    input_dim,
    output_dim,
    n_train,
    batch_size,
    t_lim,
    num_hidden_units,
    e_lims,
    n_repeats
):
    """
    Main function for this script, wrapped by argparse for command-line
    arguments.
    """
        
    # Perform warmup experiment so process acquires priority
    optimisers.warmup()

    # Initialise data, time limit, and results list
    np.random.seed(9251)
    sin_data = data.Sinusoidal(
        input_dim=input_dim,
        output_dim=output_dim,
        n_train=n_train
    )
    t_interval = t_lim / 50
    results_list = []

    for i in range(n_repeats):
        # Set the random seed
        np.random.seed(i)
        # Generate random network
        n = models.NeuralNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            num_hidden_units=num_hidden_units,
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
            batch_getter=optimisers.batch.ConstantBatchSize(batch_size)
        )
        results_list.append(result)
        
        # Generate random network
        n = models.NeuralNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            num_hidden_units=num_hidden_units,
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
            batch_getter=optimisers.batch.ConstantBatchSize(batch_size)
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
        e_lims=e_lims,
        tp=0.5
    )


if __name__ == "__main__":
    
    # Define CLI using argparse
    parser = ArgumentParser(
        description="Compare different initialisers for neural networks"
    )

    parser.add_argument(
        "input_dim",
        help="Number of input dimensions",
        default=2,
        type=int
    )
    parser.add_argument(
        "output_dim",
        help="Number of output dimensions",
        default=3,
        type=int
    )
    parser.add_argument(
        "n_train",
        help="Number of points in the training set",
        default=2500,
        type=int
    )
    parser.add_argument(
        "batch_size",
        help="Batch size to use for training",
        default=50,
        type=int
    )
    parser.add_argument(
        "t_lim",
        help="Length of time to train for each experiment",
        default=5,
        type=float
    )
    parser.add_argument(
        "num_hidden_units",
        help="Comma-separated list of hidden units per layer, EG 4,5,6",
        default="10",
        type=str
    )
    parser.add_argument(
        "e_lo",
        help="Lower axis limits for output plot",
        default=0,
        type=float
    )
    parser.add_argument(
        "e_hi",
        help="Upper axis limits for output plot",
        default=0.2,
        type=float
    )
    parser.add_argument(
        "n_repeats",
        help="Number of repeats to perform of each experiment",
        default=3,
        type=int
    )

    # Parse arguments
    args = parser.parse_args()

    # Convert comma-separated string to list of ints
    num_hidden_units = [int(i) for i in args.num_hidden_units.split(",")]

    # Call main function using command-line arguments
    t_start = perf_counter()
    main(
        args.input_dim,
        args.output_dim,
        args.n_train,
        args.batch_size,
        args.t_lim,
        num_hidden_units,
        [args.e_lo, args.e_hi],
        args.n_repeats,
    )
    print("Main function run in %.3f s" % (perf_counter() - t_start))
