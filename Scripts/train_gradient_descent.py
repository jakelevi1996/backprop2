""" This script will train a model on a dataset using gradient descent, and plot
the resulting learning curve. Using a line-search during optimisation is
optional (by default, a line-search will be used). Optionally also plot:
-   The final predictions of the model for the dataset
-   A gif of the predictions and hidden layer activations evolving over time

Below are some examples for calling this script:

    python Scripts/train_gradient_descent.py -i1 -o1 --plot_preds

    python Scripts/train_gradient_descent.py -i2 -o3 -n2500 -b200 -u 20,20 -t10 --plot_preds

To get help information for the available arguments, use the following command:

    python Scripts/train_gradient_descent.py -h

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
    error_lims,
    n_repeats,
    line_search,
    t_eval,
    plot_preds,
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
    -   error_lims: list of 2 floats, used as axis limits for the error function
        axes in the output plots, or None, in which case the axis limits are
        automatically calculated
    -   n_repeats: positive integer number of repeats to perform of each
        experiment
    -   line_search: TODO
    -   t_eval: TODO
    -   plot_preds: TODO
    """
    np.random.seed(1913)

    # Get output directory which is specific to the script parameters
    param_str = " ".join([
        "i%s"   % input_dim,
        "o%s"   % output_dim,
        "t%s"   % t_lim,
        "n%s"   % n_train,
        "b%s"   % batch_size,
        "u%s"   % num_hidden_units,
    ])
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(
        current_dir,
        "Outputs",
        "Train gradient descent",
        param_str
    )

    # Perform warmup experiment
    optimisers.warmup()

    result_list = []
    model_list = []

    dataset = data.Sinusoidal(input_dim, output_dim, n_train)

    for _ in range(n_repeats):
        model = models.NeuralNetwork(input_dim, output_dim, num_hidden_units)

        result = optimisers.Result()

        if line_search is not None:
            line_search_col = optimisers.results.columns.StepSize(line_search)
            result.add_column(line_search_col)
        
        optimisers.gradient_descent(
            model,
            dataset,
            line_search=line_search,
            result=result,
            evaluator=optimisers.Evaluator(t_interval=t_eval),
            terminator=optimisers.Terminator(t_lim=t_lim),
            batch_getter=optimisers.batch.ConstantBatchSize(batch_size, False)
        )

        result_list.append(result)
        model_list.append(model)
    
    # Make output plots
    print("Plotting output plots in \"%s\"..." % output_dir)
    plotting.plot_training_curves(
        result_list,
        dir_name=output_dir,
        e_lims=error_lims
    )
    for i, model in enumerate(model_list):
        output_dir_repeat = os.path.join(output_dir, "Repeat %i" % (i + 1))
        if plot_preds:
            plot_name = "Final predictions"
            if input_dim == 1:
                plotting.plot_1D_regression(
                    plot_name,
                    output_dir_repeat,
                    dataset,
                    model,
                )
            elif input_dim == 2:
                x_pred = lambda d: np.linspace(
                    min(dataset.x_test[d, :]),
                    max(dataset.x_test[d, :]
                ))
                plotting.plot_2D_nD_regression(
                    plot_name,
                    output_dir_repeat,
                    output_dim,
                    dataset,
                    x_pred(0),
                    x_pred(1),
                    model
                )
            else:
                raise ValueError(
                    "Can only plot predictions when the input dimension is 1 "
                    "or 2"
                )


if __name__ == "__main__":
    
    # Define CLI using argparse
    parser = ArgumentParser(
        description="train a model on a dataset using gradient descent, and "
        "plot the resulting learning curve"
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
        "-b",
        "--batch_size",
        help="Batch size to use for training, default is 50",
        default=50,
        type=int
    )
    parser.add_argument(
        "-t",
        "--t_lim",
        help="Length of time to train for each experiment in seconds. Default "
        "is 5 seconds",
        default=5,
        type=float
    )
    parser.add_argument(
        "--t_eval",
        help="Length of time between each time the model is evaluated during "
        "optimisation in seconds. Default is t_lim / 50",
        default=None,
        type=float
    )
    parser.add_argument(
        "-u",
        "--num_hidden_units",
        help="Comma-separated list of hidden units per layer, EG 4,5,6",
        default="10",
        type=str
    )
    parser.add_argument(
        "-e",
        "--error_lims",
        help="Comma-separated list of 2 floats describing the limits to use "
        "for the axes representing the error function in the output plots. "
        "Negative numbers should be prefixed with the character 'n' instead of "
        "a negative sign, so that this value is not confused with another "
        "command-line argument, EG \"n0.05,0.05\". Default is "
        "automatically-calculated axis limits.",
        default=None,
        type=str
    )
    parser.add_argument(
        "-r",
        "--n_repeats",
        help="Number of repeats to perform of each experiment",
        default=3,
        type=int
    )
    parser.add_argument(
        "--no_line_search",
        help="If this flag is included, then no line-search is used during "
        "optimisation",
        action="store_true",
    )
    parser.add_argument(
        "--plot_preds",
        help="If this flag is included, then after training has finished, plot "
        "the final predictions of the model on the data-set (only valid for 1D "
        "or 2D inputs)",
        action="store_true",
    )
    parser.add_argument(
        "--plot_pred_gif",
        help="If this flag is included, then after training has finished, plot "
        "a gif of the predictions of the model on the data-set evolving during "
        "training (only valid for 1D or 2D inputs)",
        action="store_true",
    )

    # Parse arguments
    args = parser.parse_args()

    num_hidden_units = [int(i) for i in args.num_hidden_units.split(",")]
    
    line_search = None if args.no_line_search else optimisers.LineSearch()

    args.t_eval = args.t_lim / 50 if args.t_eval is None else args.t_eval

    if args.error_lims is not None:
        float_fmt = lambda e: -float(e[1:]) if e.startswith("n") else float(e)
        args.error_lims = [float_fmt(e) for e in args.error_lims.split(",")]
        error_msg = "Must provide 2 comma-separated values for error_lims"
        assert len(args.error_lims) == 2, error_msg

    # Call main function using command-line arguments
    t_start = perf_counter()
    main(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        n_train=args.n_train,
        batch_size=args.batch_size,
        t_lim=args.t_lim,
        num_hidden_units=num_hidden_units,
        error_lims=args.error_lims,
        n_repeats=args.n_repeats,
        line_search=line_search,
        t_eval=args.t_eval,
        plot_preds=args.plot_preds
    )
    print("Main function run in %.3f s" % (perf_counter() - t_start))
