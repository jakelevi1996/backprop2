""" Make gif which plots the reduction in error function vs batch size at
regular stages during training (each frame in the gif, corresponding to a
different iteration during training, is also saved as an individual image).

TODO:
-   Put output files into parameter-specific subfolder (NB: kwarg dict can be
    formatted into a string using the expression `", ".join("%i = %i" % (k, v)
    for k, v in kwarg.items())`)
-   Add red circle and vertical line in right hand subplot for the best mean
    ratio of reduction to batch size, and plot the optimum batch size throughout
    training in separate plot.
-   Test 2 dimensional sine data (first need to determine sensible parameters
    from the test; will need to run other scripts to plot learning curve to
    determine this)

Below are some examples for calling this script:

    python Scripts\plot_error_reduction_vs_batch_size_gif.py

    python Scripts\plot_error_reduction_vs_batch_size_gif.py --no_replace

    python Scripts\plot_error_reduction_vs_batch_size_gif.py -n100 -b50 --ylims "n1e-3,1e-3,n1e-5,1e-5" --n_plots 50

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
        be an iteratble containing 4 floats, the first 2 are the lower and upper
        axis limits for the left subplot, and the second 2 are the lower and
        upper axis limits for the right subplot
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

    # Initialise models, data, and list of batch sizes to test
    model = models.NeuralNetwork(input_dim, output_dim, num_hidden_units)
    freq = 1 if (input_dim == 1) else None
    sin_data = data.Sinusoidal(input_dim, output_dim, n_train, freq=freq)
    batch_size_list = np.linspace(min_batch_size, n_train, n_batch_sizes)
    
    # Initialise objects to use during optimisation and testing batch sizes
    result_optimise = optimisers.Result()
    result_test_batch = optimisers.Result(
        verbose=False,
        add_default_columns=False
    )
    evaluator = optimisers.DoNotEvaluate()
    if batch_size_optimise is None:
        batch_getter_optimise = optimisers.batch.FullTrainingSet()
    else:
        batch_getter_optimise = optimisers.batch.ConstantBatchSize(
            batch_size_optimise,
            use_replacement
        )
    terminator_optimise = optimisers.Terminator(i_lim=n_iters_per_plot)
    terminator_test_batch = optimisers.Terminator(i_lim=1)
    line_search_optimise = optimisers.LineSearch()
    line_search_test_batch = optimisers.LineSearch()

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
    frame_dir = os.path.join(output_dir, "Frames")
    # Initialise lists for filenames and results
    filename_list = []
    best_batch_size_list = []
    best_reduction_rate_list = []

    # Iterate through the frames in the gif
    for plot_num in range(n_plots):
        # Do initial set of iterations
        optimisers.gradient_descent(
            model,
            sin_data,
            batch_getter=batch_getter_optimise,
            terminator=terminator_optimise,
            evaluator=evaluator,
            result=result_optimise,
            display_summary=False,
            line_search=line_search_optimise
        )

        # Get parameters and current test error
        w_0 = model.get_parameter_vector().copy()
        model.forward_prop(sin_data.x_test)
        E_0 = model.mean_error(sin_data.y_test)
        reduction_dict = dict()
        # Iterate through batch sizes
        for batch_size in batch_size_list:
            # Set number of repeats and initialise results list
            reduction_dict[batch_size] = []
            batch_getter = optimisers.batch.ConstantBatchSize(
                int(batch_size),
                replace=use_replacement
            )
            # Iterate through repeats of the batch size
            for _ in range(n_repeats):
                # Perform one iteration of gradient descent
                line_search_test_batch.s = line_search_optimise.s
                optimisers.gradient_descent(
                    model,
                    sin_data,
                    batch_getter=batch_getter,
                    terminator=terminator_test_batch,
                    evaluator=evaluator,
                    result=result_test_batch,
                    display_summary=False,
                    line_search=line_search_test_batch
                )
                # Calculate new error and add the reduction to the list
                model.forward_prop(sin_data.x_test)
                E_new = model.mean_error(sin_data.y_test)
                error_reduction = E_0 - E_new
                reduction_dict[batch_size].append(error_reduction)
                # Reset parameters
                model.set_parameter_vector(w_0.copy())
            print(".", end="", flush=True)

        # Make plot of error reduction vs batch size
        Iteration = optimisers.results.columns.Iteration
        last_iter = result_optimise.get_values(Iteration)[-1]
        title = "Error reduction vs batch size, iteration = %05i" % last_iter
        (
            full_path,
            best_batch_size,
            best_reduction_rate
        ) = plotting.plot_error_reductions_vs_batch_size(
            title,
            frame_dir,
            reduction_dict,
            y_lim_left=ylims[:2],
            y_lim_right=ylims[2:]
        )
        # Store the filename, and best batch and reduction over batch
        filename_list.append(full_path)
        best_batch_size_list.append(best_batch_size)
        best_reduction_rate_list.append(best_reduction_rate)
        print("")
    
    # Make gif out of the inidividual image frames
    frame_duration_ms = 1000 * gif_duration / n_plots
    plotting.make_gif(
        "Error reduction vs batch size",
        output_dir,
        filename_list,
        duration=frame_duration_ms,
        loop=None
    )
    plotting.plot_training_curves([result_optimise], dir_name=output_dir)
    plotting.plot_optimal_batch_sizes(
        "Optimal batch size",
        output_dir,
        best_batch_size_list,
        best_reduction_rate_list,
        result_optimise,
    )
    

if __name__ == "__main__":
    
    # Define CLI using argparse
    parser = ArgumentParser(description=
        "Make gif which plots the reduction in error function vs batch size "
        "at regular stages during training"
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
        help=(
            "Comma separated list of 4 floats describing the limits to use for "
            "the y axes. Negative numbers should be prefixed with the "
            "character 'n' instead of a negative sign, so that this value is "
            "not confused with another command-line argument. See main "
            "function docstring for more info"
        ),
        default="n0.05,0.05,n0.01,0.01",
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
        help=(
            "Batch size to use for standard optimisation iterations (IE not "
            "when sweeping over batch sizes). If ommitted, then the full "
            "training set is used as a batch during optimisation iterations"
        ),
        default=None,
        type=int
    )
    parser.add_argument(
        "--no_replace",
        help=(
            "Don't use replacement when sampling batches from the training set"
        ),
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
    float_fmt = lambda y: -float(y[1:]) if y.startswith("n") else float(y)
    ylims = [float_fmt(f) for f in args.ylims.split(",")]
    assert len(ylims) == 4, "Must provide 4 comma-separated values for ylims"

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
        ylims,
        args.seed,
        args.batch_size_optimise,
        use_replacement,
        args.gif_duration
    )
    print("Main function run in %.3f s" % (perf_counter() - t_start))
