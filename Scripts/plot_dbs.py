"""
Script to DBS metric and batch size, when training a neural network on
sinusoidal data using gradient descent with a line-search and a dynamic batch
size.

This script requires command line arguments for the input and output dimensions
and number of training points in the data-set, the length of time to train for
each experiment, the number of hidden units in each hidden layer of the
NeuralNetwork, the axis limits in the output plots, and the number of repeats to
perform of each experiment.

Below are some examples for calling this script:

    python Scripts\plot_dbs.py 1 1 100 1 10 0 0.02 3 0.995 0.99 10

    python Scripts\plot_dbs.py 2 3 2500 20 20,20 0 4 3 0.995 0.99999 10

Running each of the above examples requires ??? s and ??? s respectively.

To get help information for the available arguments, use the following command:

    python Scripts\plot_dbs.py -h

"""
import os
from argparse import ArgumentParser
from time import perf_counter
import numpy as np
if __name__ == "__main__":
    import __init__
import data, models, optimisers, plotting

# Get name of output directory, and create directory if it doesn't exist
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs", "DBS experiments")
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

def main(
    input_dim,
    output_dim,
    n_train,
    t_lim,
    num_hidden_units,
    e_lims,
    n_repeats,
    alpha_smooth,
    p_c,
    min_batch_size
):
    """
    Main function for the script. See module docstring for more info.

    Inputs:
    -   input_dim: positive integer number of input dimensions
    -   output_dim: positive integer number of output dimensions
    -   n_train: positive integer number of points in the training set
    -   t_lim: positive float, length of time to train for each experiment
    -   num_hidden_units: list of positive integers, number of hidden units in
        each hidden layer of the NeuralNetwork, EG [10] or [20, 20]
    -   e_lims: list of 2 floats, used as axis limits in the output plots
    -   n_repeats: positive integer number of repeats to perform of each
        experiment
    -   alpha_smooth: float in (0, 1), amount of smoothing to apply to DBS batch
        size
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

    for i in range(n_repeats):
        # Set the random seed
        np.random.seed(i)
        # Generate random network
        model = models.NeuralNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            num_hidden_units=num_hidden_units,
            act_funcs=[models.activations.gaussian, models.activations.identity]
        )
        # Call gradient descent function
        result = optimisers.Result("Repeat = %i" % i)
        batch_getter = optimisers.batch.DynamicBatchSize(
            model,
            sin_data,
            alpha_smooth=alpha_smooth,
            prob_correct_direction=p_c,
            min_batch_size=min_batch_size
        )
        result.add_column(optimisers.results.columns.BatchSize(batch_getter))
        result.add_column(optimisers.results.columns.DbsMetric())
        result.add_column(optimisers.results.columns.GlobalDbsMetric())
        result = optimisers.gradient_descent(
            model,
            sin_data,
            terminator=optimisers.Terminator(t_lim=t_lim),
            evaluator=optimisers.Evaluator(t_interval=t_interval),
            result=result,
            line_search=optimisers.LineSearch(),
            batch_getter=batch_getter
        )
        results_list.append(result)

    # Compare training curves
    plot_name_suffix = "\n%iD-%iD data"             % (input_dim, output_dim)
    plot_name_suffix += ", %.2g s training time"    % t_lim
    plot_name_suffix += ", %s hidden units"         % str(num_hidden_units)
    plot_name_suffix += "\nalpha_smooth = %.3f"     % alpha_smooth
    plot_name_suffix += ", p_c = %.3f"              % p_c
    plot_name_suffix += ", min_batch_size = %.3f"   % min_batch_size
    this_test_output_dir = os.path.join(
        output_dir,
        plot_name_suffix.replace("\n", "")
    )
    plotting.plot_training_curves(
        results_list,
        "DBS learning curves" + plot_name_suffix,
        this_test_output_dir,
        e_lims=e_lims
    )
    for attr_name in ["dbs_metric", "global_dbs_metric", "batch_size"]:
        plot_name = "%s against iteration for dynamic batch size" % attr_name
        plot_name += plot_name_suffix
        plotting.plot_result_attribute(
            plot_name,
            this_test_output_dir,
            results_list,
            attr_name
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
    parser.add_argument(
        "alpha_smooth",
        help="Amount of smoothing to apply to DBS batch size",
        default=0.99,
        type=float
    )
    parser.add_argument(
        "p_c",
        help="Probability of moving in the correct direction",
        default=0.99,
        type=float
    )
    parser.add_argument(
        "min_batch_size",
        help="Minimum batch size",
        default=10,
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
        args.t_lim,
        num_hidden_units,
        [args.e_lo, args.e_hi],
        args.n_repeats,
        args.alpha_smooth,
        args.p_c,
        args.min_batch_size,
    )
    print("Main function run in %.3f s" % (perf_counter() - t_start))
