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

    python Scripts\plot_error_reduction_vs_batch_size_gif.py -i1 -o1 -n100 -b50 -t1 -u10 -l0 -g 0.02 -r3

    python Scripts\plot_error_reduction_vs_batch_size_gif.py -i2 -o3 -n2500 -b50 -t10 -u 20,20 -l0 -g4 -r3

Running each of the above examples requires 12.858 s and 121.004 s respectively.

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
    seed
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
    -   n_plots: number of frames of the gif
    -   n_batch_sizes: the number of different batch sizes to test for each
        iteration
    -   min_batch_size: the smallest batch size to test
    -   ylims: limits for the y-axes of each subplot of the output gif. Should
        be an iteratble containing 4 floats, the first 2 are the lower and upper
        axis limits for the left subplot, and the second 2 are the lower and
        upper axis limits for the right subplot
    -   seed: random seed to use for the experiment
    """
    np.random.seed(seed)
    n_iters_per_plot = int(n_iters / n_plots)

    model = models.NeuralNetwork(input_dim, output_dim, num_hidden_units)
    sin_data = data.Sinusoidal(input_dim, output_dim, n_train, freq=1)

    batch_size_list = np.linspace(min_batch_size, n_train, n_batch_sizes)
    
    result_optimise = optimisers.Result()
    result_test_batch = optimisers.Result(
        verbose=False,
        add_default_columns=False
    )
    evaluator = optimisers.DoNotEvaluate()
    batch_getter_optimise = optimisers.batch.FullTrainingSet(sin_data)
    terminator_optimise = optimisers.Terminator(i_lim=n_iters_per_plot)
    terminator_test_batch = optimisers.Terminator(i_lim=1)
    line_search_optimise = optimisers.LineSearch()
    line_search_test_batch = optimisers.LineSearch()

    # Get output dir (TODO: make specific to the script parameters)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "Outputs", "Error vs batch")
    filename_list = []


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
        reduction_list_list = []
        # Iterate through batch sizes
        for batch_size in batch_size_list:
            # Set number of repeats and initialise results list
            reduction_list_list.append([])
            batch_getter = optimisers.batch.ConstantBatchSize(int(batch_size))
            # Iterate through repeats
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
                # Calculate new error and add the reduction to the list TODO:
                # could this test set error be taken from result_test_batch?
                model.forward_prop(sin_data.x_test)
                error_reduction = E_0 - model.mean_error(sin_data.y_test)
                reduction_list_list[-1].append(error_reduction)
                # Reset parameters
                model.set_parameter_vector(w_0)
            print(".", end="", flush=True)

        # Make plot of batch size vs gif
        title = "Error reduction vs batch size, iteration = %05i" % (
            result_optimise.get_values("iteration")[-1]
        )
        full_path = plotting.plot_error_reductions_vs_batch_size(
            title,
            output_dir,
            batch_size_list,
            reduction_list_list,
            y_lim_left=ylims[:2],
            y_lim_right=ylims[2:]
        )
        filename_list.append(full_path)
        print("")
    
    plotting.make_gif(
        "Error reduction vs batch size",
        output_dir,
        filename_list,
        duration=500
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
            "Comma separated list of floats describing the limits to use for "
            "the y axes; see main function docstring for more info",
        ),
        default="-0.05,0.05,-0.01,0.01",
        type=str
    )
    parser.add_argument(
        "--seed",
        help="Random seed to use for the experiment",
        default=1913,
        type=int
    )

    # Parse arguments
    args = parser.parse_args()

    # Convert comma-separated string to list of ints
    num_hidden_units = [int(i) for i in args.num_hidden_units.split(",")]
    ylims = [float(f) for f in args.ylims.split(",")]

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
        args.seed
    )
    print("Main function run in %.3f s" % (perf_counter() - t_start))
