""" Make gif which plots the reduction in error function vs batch size at
regular stages during training (each frame in the gif, corresponding to a
different iteration during training, is also saved as an individual image).

TODO:
-   Test 2 dimensional sine data (first need to determine sensible parameters
    from the test; will need to run other scripts to plot learning curve to
    determine this)

Below are some examples for calling this script:

    python Scripts\plot_error_reduction_vs_batch_size_gif.py

    python Scripts\plot_error_reduction_vs_batch_size_gif.py --no_replace

    python Scripts\plot_error_reduction_vs_batch_size_gif.py -n100 -b50 --n_plots 50

    python Scripts\plot_error_reduction_vs_batch_size_gif.py -n300 -b100

    python Scripts\plot_error_reduction_vs_batch_size_gif.py -i2 -o3 -n2500 -u 20,20

To get help information for the available arguments, use the following command:

    python Scripts\plot_error_reduction_vs_batch_size_gif.py -h

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
    num_hidden_units,
    n_repeats,
    n_iters,
    n_plots,
    n_batch_sizes,
    min_batch_size,
    ylims,
    seed,
    batch_size_optimise,
    use_replacement,
    gif_duration
):
    """
    Main function for the script. See module docstring for more info.

    Inputs:
    -   input_dim: positive integer number of input dimensions
    -   output_dim: positive integer number of output dimensions
    -   n_train: positive integer number of points in the training set
    -   num_hidden_units: list of positive integers, number of hidden units in
        each hidden layer of the NeuralNetwork, EG [10] or [20, 20]
    -   n_repeats: positive integer number of repeats to perform of each batch
        size test
    -   n_iters: total number of iterations to perform
    -   n_plots: number of frames of the gif (equal to how many times
        optimisation will pause in order to sweep over the list of batch sizes)
    -   n_batch_sizes: the number of different batch sizes to test for each
        iteration
    -   min_batch_size: the smallest batch size to test
    -   ylims: limits for the y-axes of each subplot of the output gif. Should
        be None, in which case the axis limits are calculated automatically, or
        an iterable containing 4 floats, in which the first 2 are the lower and
        upper axis limits for the left subplot, and the second 2 are the lower
        and upper axis limits for the right subplot
    -   seed: random seed to use for the experiment
    -   batch_size_optimise: batch size to use for standard optimisation
        iterations (IE not when sweeping over batch sizes). If ommitted, then
        the full training set is used as a batch during optimisation iterations
    -   use_replacement: if True, then use replacement when sampling batches
        from the training set
    -   gif_duration: time in seconds that the output gif should last for in
        total
    """
    np.random.seed(seed)
    n_iters_per_plot = int(n_iters / n_plots)

    # Initialise model and dataset
    model = models.NeuralNetwork(input_dim, output_dim, num_hidden_units)
    freq = 1 if (input_dim == 1) else None
    sin_data = data.Sinusoidal(input_dim, output_dim, n_train, freq=freq)
    
    # Initialise objects for optimisation
    result = optimisers.Result()
    evaluator = optimisers.Evaluator(i_interval=n_iters_per_plot)
    terminator = optimisers.Terminator(i_lim=n_iters)
    if batch_size_optimise is None:
        batch_getter = optimisers.batch.FullTrainingSet()
    else:
        batch_getter = optimisers.batch.ConstantBatchSize(
            batch_size_optimise,
            use_replacement
        )
    line_search = optimisers.LineSearch()

    # Initialise OptimalBatchSize column and add to the result object
    optimal_batch_size_col = optimisers.results.columns.OptimalBatchSize(
        sin_data.n_train,
        optimisers.gradient_descent,
        line_search,
        n_repeats=n_repeats,
        n_batch_sizes=n_batch_sizes
    )
    result.add_column(optimal_batch_size_col)

    # Get output directory which is specific to the script parameters
    param_str = ", ".join([
        "input_dim = %i"            % input_dim,
        "output_dim = %i"           % output_dim,
        "n_train = %i"              % n_train,
        "n_iters = %i"              % n_iters,
        "batch_size_optimise = %r"  % batch_size_optimise,
        "use_replacement = %r"      % use_replacement,
        "ylims = %r"                % ylims,
        "n_plots = %i"              % n_plots,
    ])
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(
        current_dir,
        "Outputs",
        "Error vs batch",
        param_str
    )

    # Call optimisation function
    optimisers.gradient_descent(
        model,
        sin_data,
        result=result,
        batch_getter=batch_getter,
        terminator=terminator,
        evaluator=evaluator,
        line_search=line_search
    )

    # Make output plots
    print("Plotting output plots in \"%s\"..." % output_dir)
    plotting.plot_training_curves([result], dir_name=output_dir)
    frame_duration_ms = 1000 * gif_duration / n_plots
    if ylims is None:
        y_lim_left = None
        y_lim_right = None
    else:
        y_lim_left = ylims[:2]
        y_lim_right = ylims[2:]
    plotting.plot_error_reductions_vs_batch_size_gif(
        result,
        optimal_batch_size_col,
        output_dir,
        y_lim_left=y_lim_left,
        y_lim_right=y_lim_right,
        duration=frame_duration_ms,
        loop=None
    )
    plotting.plot_optimal_batch_sizes(
        "Optimal batch size",
        output_dir,
        result,
        optimal_batch_size_col,
    )


if __name__ == "__main__":
    
    # Define CLI using argparse
    parser = ArgumentParser(
        description="Make gif which plots the reduction in error function vs "
        "batch size at regular stages during training"
    )

    parser.add_argument(
        "-i",
        "--input_dim",
        help="Number of input dimensions",
        default=1,
        type=int
    )
    parser.add_argument(
        "-o",
        "--output_dim",
        help="Number of output dimensions",
        default=1,
        type=int
    )
    parser.add_argument(
        "-n",
        "--n_train",
        help="Number of points in the training set",
        default=100,
        type=int
    )
    parser.add_argument(
        "-u",
        "--num_hidden_units",
        help="Comma-separated list of hidden units per layer, EG 4,5,6",
        default="10",
        type=str
    )
    parser.add_argument(
        "-r",
        "--n_repeats",
        help="Number of repeats to perform of each batch size test",
        default=100,
        type=int
    )
    parser.add_argument(
        "--n_iters",
        help="Total number of iterations to perform",
        default=10000,
        type=int
    )
    parser.add_argument(
        "--n_plots",
        help="Number of frames of the gif",
        default=20,
        type=int
    )
    parser.add_argument(
        "--n_batch_sizes",
        help="The number of different batch sizes to test for each iteration",
        default=30,
        type=int
    )
    parser.add_argument(
        "--min_batch_size",
        help="The smallest batch size to test",
        default=5,
        type=int
    )
    parser.add_argument(
        "--ylims",
        help="Comma separated list of 4 floats describing the limits to use "
        "for the y axes. Negative numbers should be prefixed with the "
        "character 'n' instead of a negative sign, so that this value is not "
        "confused with another command-line argument, EG "
        "\"n0.05,0.05,n0.01,0.01\". Default is automatically-calculated axis "
        "limits. See main function docstring for more info",
        default=None,
        type=str
    )
    parser.add_argument(
        "--seed",
        help="Random seed to use for the experiment",
        default=1913,
        type=int
    )
    parser.add_argument(
        "-b",
        "--batch_size_optimise",
        help="Batch size to use for standard optimisation iterations (IE not "
        "when sweeping over batch sizes). If ommitted, then the full training "
        "set is used as a batch during optimisation iterations",
        default=None,
        type=int
    )
    parser.add_argument(
        "--no_replace",
        help="Don't use replacement when sampling batches from the training "
        "set",
        action="store_true"
    )
    parser.add_argument(
        "--gif_duration",
        help="Time in seconds that the output gif should last for in total",
        default=20,
        type=float
    )

    # Parse arguments
    args = parser.parse_args()
    use_replacement = not args.no_replace
    num_hidden_units = [int(i) for i in args.num_hidden_units.split(",")]
    if args.ylims is not None:
        float_fmt = lambda y: -float(y[1:]) if y.startswith("n") else float(y)
        args.ylims = [float_fmt(f) for f in args.ylims.split(",")]
        error_msg = "Must provide 4 comma-separated values for ylims"
        assert len(args.ylims) == 4, error_msg

    # Call main function using command-line arguments
    t_start = perf_counter()
    main(
        args.input_dim,
        args.output_dim,
        args.n_train,
        num_hidden_units,
        args.n_repeats,
        args.n_iters,
        args.n_plots,
        args.n_batch_sizes,
        args.min_batch_size,
        args.ylims,
        args.seed,
        args.batch_size_optimise,
        use_replacement,
        args.gif_duration
    )
    print("Main function run in %.3f s" % (perf_counter() - t_start))
