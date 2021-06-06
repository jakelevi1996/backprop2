""" This script will train a model on a dataset using gradient descent, and
plot the resulting learning curve. Using a line-search during optimisation is
optional (by default, a line-search will be used). Optionally also plot:
-   The final predictions of the model for the dataset
-   A gif of the predictions evolving over time
-   A gif of the hidden layer activations evolving over time

Below are some examples for calling this script:

    python Scripts/train_gradient_descent.py -i1 -o1 --plot_preds --plot_pred_gif --plot_hidden_gif

    python Scripts/train_gradient_descent.py -i2 -o3 -n2500 -b200 -u 20,20 -t10 --plot_preds

    python Scripts/train_gradient_descent.py -i2 -o10 -n2500 -b200 -u 20,20 -t2 --plot_preds -dMixtureOfGaussians

    python Scripts/train_gradient_descent.py -i2 -n2500 -b200 -u 20,20 -t1 --plot_preds --plot_pred_gif -dBinaryMixtureOfGaussians

    python Scripts/train_gradient_descent.py -i2 -n1000 -b200 -u 20,20 -t1 --plot_preds -dXor

    python Scripts/train_gradient_descent.py -i2 -n1000 -b200 -u 20,20 -t1 --plot_preds -dDisk

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
    plot_pred_gif,
    plot_hidden_gif,
    plot_hidden_preactivations_gif,
    dataset_type,
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
    -   plot_pred_gif: TODO
    -   plot_hidden_gif: TODO
    -   plot_hidden_preactivations_gif: TODO
    -   dataset_type: TODO
    """
    np.random.seed(1913)

    # Get output directory which is specific to the script parameters
    param_str = " ".join([
        "d%s"   % dataset_type.__name__[:3],
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

    # Initialise lists of objects that will be stored for each repeat
    result_list = []
    model_list = []
    prediction_column_list = []

    # Initialise dataset object and corresponding error function
    dataset_kwargs = {"input_dim": input_dim, "n_train": n_train}
    if not issubclass(dataset_type, data.BinaryClassification):
        dataset_kwargs["output_dim"] = output_dim
    dataset = dataset_type(**dataset_kwargs)

    if isinstance(dataset, data.Regression):
        error_func = models.errors.sum_of_squares
        act_funcs = None
        print("Using regression data set with sum of squares error function")
    elif isinstance(dataset, data.BinaryClassification):
        error_func = models.errors.binary_cross_entropy
        act_funcs = [models.activations.gaussian, models.activations.logistic]
        print(
            "Using binary classification data set with binary cross-entropy "
            "error function, and logistic activation function in the output "
            "layer"
        )
    elif isinstance(dataset, data.Classification):
        error_func = models.errors.softmax_cross_entropy
        act_funcs = None
        print(
            "Using classification data set with softmax cross entropy error "
            "function"
        )
    else:
        raise ValueError(
            "Data set must be either a binary-classification, multi-class "
            "classification or regression data set"
        )

    # Iterate through repeats
    for _ in range(n_repeats):
        # Initialise model and Result object
        model = models.NeuralNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            num_hidden_units=num_hidden_units,
            error_func=error_func,
            act_funcs=act_funcs,
        )

        result = optimisers.Result()

        if line_search is not None:
            line_search_col = optimisers.results.columns.StepSize(line_search)
            result.add_column(line_search_col)

        if plot_pred_gif or plot_hidden_gif:
            pred_column = optimisers.results.columns.Predictions(
                dataset=dataset,
                store_hidden_layer_outputs=plot_hidden_gif,
                store_hidden_layer_preactivations=(
                    plot_hidden_preactivations_gif
                ),
            )
            result.add_column(pred_column)

        # Perform gradient descent
        optimisers.gradient_descent(
            model,
            dataset,
            line_search=line_search,
            result=result,
            evaluator=optimisers.Evaluator(t_interval=t_eval),
            terminator=optimisers.Terminator(t_lim=t_lim),
            batch_getter=optimisers.batch.ConstantBatchSize(batch_size, False),
        )

        # Store results
        result_list.append(result)
        model_list.append(model)
        if plot_pred_gif or plot_hidden_gif:
            prediction_column_list.append(pred_column)
    
    # Make output plots
    print("Plotting output plots in \"%s\"..." % output_dir)
    plotting.plot_training_curves(
        result_list,
        dir_name=output_dir,
        e_lims=error_lims,
    )
    os.system("explorer \"%s\"" % output_dir)
    for i, model in enumerate(model_list):
        output_dir_repeat = os.path.join(output_dir, "Repeat %i" % (i + 1))
        if plot_preds:
            print("Plotting final predictions...")
            plotting.plot_data_predictions(
                plot_name="Final predictions",
                dir_name=output_dir_repeat,
                dataset=dataset,
                output_dim=output_dim,
                model=model,
            )
        if plot_pred_gif:
            print("Plotting gif of predictions during training...")
            plotting.plot_predictions_gif(
                plot_name="Model predictions during training",
                dir_name=output_dir_repeat,
                result=result_list[i],
                prediction_column=prediction_column_list[i],
                dataset=dataset,
                output_dim=output_dim,
                duration=t_eval*1000,
            )
        if plot_hidden_gif:
            print("Plotting gif of hidden layers during training...")
            if plot_hidden_preactivations_gif:
                plot_name="Hidden layer preactivations during training"
            else:
                plot_name="Hidden layer outputs during training"

            plotting.plot_hidden_outputs_gif(
                plot_name=plot_name,
                dir_name=output_dir_repeat,
                result=result_list[i],
                prediction_column=prediction_column_list[i],
                dataset=dataset,
                output_dim=output_dim,
                duration=t_eval*1000,
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
        type=int,
    )
    parser.add_argument(
        "-o",
        "--output_dim",
        help="Number of output dimensions",
        default=1,
        type=int,
    )
    parser.add_argument(
        "-n",
        "--n_train",
        help="Number of points in the training set",
        default=100,
        type=int,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="Batch size to use for training, default is 50",
        default=50,
        type=int,
    )
    parser.add_argument(
        "-t",
        "--t_lim",
        help="Length of time to train for each experiment in seconds. Default "
        "is 5 seconds",
        default=5,
        type=float,
    )
    parser.add_argument(
        "--t_eval",
        help="Length of time between each time the model is evaluated during "
        "optimisation in seconds. Default is t_lim / 50",
        default=None,
        type=float,
    )
    parser.add_argument(
        "-u",
        "--num_hidden_units",
        help="Comma-separated list of hidden units per layer, EG 4,5,6",
        default="10",
        type=str,
    )
    parser.add_argument(
        "-e",
        "--error_lims",
        help="Comma-separated list of 2 floats describing the limits to use "
        "for the axes representing the error function in the output plots. "
        "Negative numbers should be prefixed with the character 'n' instead "
        "of a negative sign, so that this value is not confused with another "
        "command-line argument, EG \"n0.05,0.05\". Default is "
        "automatically-calculated axis limits.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-r",
        "--n_repeats",
        help="Number of repeats to perform of each experiment",
        default=3,
        type=int,
    )
    parser.add_argument(
        "--no_line_search",
        help="If this flag is included, then no line-search is used during "
        "optimisation",
        action="store_true",
    )
    parser.add_argument(
        "--plot_preds",
        help="If this flag is included, then after training has finished, "
        "plot the final predictions of the model on the data-set (only valid "
        "for 1D or 2D inputs)",
        action="store_true",
    )
    parser.add_argument(
        "--plot_pred_gif",
        help="If this flag is included, then after training has finished, "
        "plot a gif of the predictions of the model on the data-set evolving "
        "during training (only valid for 1D or 2D inputs)",
        action="store_true",
    )
    parser.add_argument(
        "--plot_hidden_gif",
        help="If this flag is included, then after training has finished, "
        "plot a gif of the outputs from the hidden layers of the model "
        "evolving during training (only valid for 1D or 2D inputs)",
        action="store_true",
    )
    parser.add_argument(
        "--plot_hidden_preactivations_gif",
        help="If this flag is included, and the --plot_hidden_gif flag is "
        "included, then when plotting the gif of the hidden layers of the "
        "model during training, the hidden layer preactivations are used "
        "instead of the hidden layer outputs. If the --plot_hidden_gif flag "
        "is not included, then this flag has no effect.",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--dataset_type",
        help="Type of dataset to use. Options include regression and "
        "classification types of datasets. The error function, output layer "
        "activation function, and plotting functions will be chosen to match "
        "the dataset type. Options are: %r, default is 'Sinusoidal'" %
        list(data.dataset_class_dict.keys()),
        default="Sinusoidal",
        type=str,
        dest="dataset_type_str",
        choices=data.dataset_class_dict.keys(),
    )

    # Parse arguments
    args = parser.parse_args()

    num_hidden_units = [int(i) for i in args.num_hidden_units.split(",")]
    
    line_search = None if args.no_line_search else optimisers.LineSearch()

    args.t_eval = args.t_lim / 50 if args.t_eval is None else args.t_eval

    dataset_type = data.dataset_class_dict[args.dataset_type_str]

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
        plot_preds=args.plot_preds,
        plot_pred_gif=args.plot_pred_gif,
        plot_hidden_gif=args.plot_hidden_gif,
        plot_hidden_preactivations_gif=args.plot_hidden_preactivations_gif,
        dataset_type=dataset_type,
    )
    print("Main function run in %.3f s" % (perf_counter() - t_start))
