"""
Script to compare learning curves for parallel-block generalised Newton's method
vs gradient descent, both with and without line-search, on sinusoidal data.

This script requires command line arguments for the input and output dimensions
and number of training points in the data-set, the batch size and length of time
to train for each experiment, the number of hidden units in each hidden layer of
the NeuralNetwork, and the axis limits in the output plots.

Below are some examples for calling this script:

    python Scripts\compare_pbgn_gd.py 1 1 100 50 5 10 0 0.2

    python Scripts\compare_pbgn_gd.py 2 3 2500 50 10 20,20 0 4

Running each of the above examples requires ??? s and ??? s respectively.

To get help information for the available arguments, use the following command:

    python Scripts\compare_gd_pbgn_1d_sin.py -h

"""
import os
from argparse import ArgumentParser
from time import perf_counter
import numpy as np
if __name__ == "__main__":
    import __init__
import data, models, optimisers, plotting

def main(
    input_dim,
    output_dim,
    n_train,
    batch_size,
    t_lim,
    num_hidden_units,
    e_lims
):
    """
    Main function for the script. See module docstring for more info.

    Inputs:
    -   input_dim: positive integer number of input dimensions
    -   output_dim: positive integer number of output dimensions
    -   n_train: positive integer number of points in the training set
    -   batch_size: positive integer batch size to use for training
    -   t_lim: positive float, length of time to train for each experiment
    -   num_hidden_units: list of positive integers, number of hidden units in
        each hidden layer of the NeuralNetwork, EG [10] or [20, 20]
    -   e_lims: list of 2 floats, used as axis limits in the output plots
    """
    # Perform warmup experiment so process acquires priority
    optimisers.warmup()

    # Initialise data, results list, and time interval for evaluations
    np.random.seed(9251)
    sin_data = data.Sinusoidal(
        input_dim=input_dim,
        output_dim=output_dim,
        n_train=n_train,
    )
    results_list = []
    t_interval = t_lim / 50

    for seed in [2295, 6997, 7681]:
        # Set the random seed
        np.random.seed(seed)
        # Generate random network and store initial parameters
        n = models.NeuralNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            num_hidden_units=num_hidden_units,
            act_funcs=[models.activations.gaussian, models.activations.identity]
        )
        w0 = n.get_parameter_vector().copy()
        # Call gradient descent function
        result_gd_ls = optimisers.gradient_descent(
            n,
            sin_data,
            terminator=optimisers.Terminator(t_lim=t_lim),
            evaluator=optimisers.Evaluator(t_interval=t_interval),
            result=optimisers.Result(
                name="SGD with line search",
                verbose=True
            ),
            line_search=optimisers.LineSearch(),
            batch_getter=optimisers.batch.ConstantBatchSize(batch_size)
        )
        results_list.append(result_gd_ls)
        # Try again without line search
        n.set_parameter_vector(w0)
        result_gd_no_ls = optimisers.gradient_descent(
            n,
            sin_data,
            terminator=optimisers.Terminator(t_lim=t_lim),
            evaluator=optimisers.Evaluator(t_interval=t_interval),
            result=optimisers.Result(
                name="SGD without line search",
                verbose=True
            ),
            line_search=None,
            batch_getter=optimisers.batch.ConstantBatchSize(batch_size)
        )
        results_list.append(result_gd_no_ls)
        # Call generalised Newton function
        n.set_parameter_vector(w0)
        result_pbgn_ls = optimisers.generalised_newton(
            n,
            sin_data,
            terminator=optimisers.Terminator(t_lim=t_lim),
            evaluator=optimisers.Evaluator(t_interval=t_interval),
            result=optimisers.Result(
                name="PBGN with line search",
                verbose=True
            ),
            line_search=optimisers.LineSearch(),
            batch_getter=optimisers.batch.ConstantBatchSize(batch_size)
        )
        results_list.append(result_pbgn_ls)
        # Try again without line search
        n.set_parameter_vector(w0)
        result_pbgn_no_ls = optimisers.generalised_newton(
            n,
            sin_data,
            terminator=optimisers.Terminator(t_lim=t_lim),
            evaluator=optimisers.Evaluator(t_interval=t_interval),
            result=optimisers.Result(
                name="PBGN without line search",
                verbose=True
            ),
            line_search=None,
            batch_getter=optimisers.batch.ConstantBatchSize(batch_size)
        )
        results_list.append(result_pbgn_no_ls)

    # Get name of output directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "Outputs")

    # Compare training curves
    plot_name = "Comparing gradient descent vs generalised Newton"
    plot_name += ", %iD-%iD data" % (input_dim, output_dim)
    plot_name += ", %.2g s training time" % t_lim
    plot_name += ", %s hidden units" % str(num_hidden_units)
    plotting.plot_training_curves(
        results_list,
        plot_name,
        output_dir,
        e_lims=e_lims
    )

if __name__ == "__main__":
    
    # Define CLI using argparse
    parser = ArgumentParser(
        description="Compare generalised Newton's method vs gradient descent"
    )

    parser.add_argument(
        "input_dim",
        help="Number of input dimensions",
        default=1,
        type=int
    )
    parser.add_argument(
        "output_dim",
        help="Number of output dimensions",
        default=1,
        type=int
    )
    parser.add_argument(
        "n_train",
        help="Number of points in the training set",
        default=100,
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
        [args.e_lo, args.e_hi]
    )
    print("Main function run in %.3f s" % (perf_counter() - t_start))
