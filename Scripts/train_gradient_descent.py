""" This script will train a model on a dataset using gradient descent, and
plot the resulting learning curve. Using a line-search during optimisation is
optional (by default, a line-search will be used). Optionally also plot:
-   The final predictions of the model for the dataset
-   A gif of the predictions evolving over time
-   A gif of the hidden layer activations evolving over time

Below are some examples for calling this script:

    python Scripts/train_gradient_descent.py -i1 -o1 --plot_preds --plot_pred_gif --plot_hidden_gif

    python Scripts/train_gradient_descent.py -i1 -o1 --plot_preds --plot_test_set_improvement_probability

    python Scripts/train_gradient_descent.py -i1 -o1 --plot_preds --plot_test_set_improvement_probability --dynamic_terminator

    python Scripts/train_gradient_descent.py -i2 -o3 -n2500 -b200 -u 20,20 -t10 --plot_preds --plot_test_set_improvement_probability

    python Scripts/train_gradient_descent.py -i2 -o3 -n2500 -b200 -u 20,20 -t10 --plot_preds --plot_test_set_improvement_probability --dynamic_terminator

    python Scripts/train_gradient_descent.py -i2 -o3 -n2500 -b200 -u 20,20 -t10 --plot_preds --plot_test_set_improvement_probability --dynamic_terminator --dt_buffer_length 200

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
from optimisers.results import columns

def main(args):
    """
    Main function for the script. See module docstring for more info.

    Inputs:
    -   args: object containing modified command line arguments as attributes
    """
    np.random.seed(1913)

    # Get output directory which is specific to the relevant script parameters
    param_str = " ".join([
        "d%s"   % args.dataset_type.__name__[:3],
        "i%s"   % args.input_dim,
        "o%s"   % args.output_dim,
        "t%s"   % args.t_lim,
        "n%s"   % args.n_train,
        "b%s"   % args.batch_size,
        "u%s"   % args.num_hidden_units,
    ])
    if args.dynamic_terminator:
        param_str += " dyn%i" % args.dt_buffer_length

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(
        current_dir,
        "Outputs",
        "Train gradient descent",
        param_str,
    )
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Perform warmup experiment
    optimisers.warmup()

    # Initialise lists of objects that will be stored for each repeat
    result_list = []
    model_list = []
    prediction_column_list = []

    # Initialise dataset object and corresponding error function
    dataset_kwargs = {
        "input_dim":   args.input_dim,
        "n_train":     args.n_train,
    }
    if not issubclass(args.dataset_type, data.BinaryClassification):
        dataset_kwargs["output_dim"] = args.output_dim
    dataset = args.dataset_type(**dataset_kwargs)

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
    for i in range(args.n_repeats):
        # Initialise model and Result object
        model = models.NeuralNetwork(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            num_hidden_units=args.num_hidden_units,
            error_func=error_func,
            act_funcs=act_funcs,
        )

        result = optimisers.Result(name="Repeat %i" % (i + 1))

        if args.line_search is not None:
            args.line_search_col = columns.StepSize(args.line_search)
            result.add_column(args.line_search_col)

        if args.plot_pred_gif or args.plot_hidden_gif:
            pred_column = columns.Predictions(
                dataset=dataset,
                store_hidden_layer_outputs=args.plot_hidden_gif,
                store_hidden_layer_preactivations=(
                    args.plot_hidden_preactivations_gif
                ),
            )
            result.add_column(pred_column)
        
        if args.plot_test_set_improvement_probability:
            test_set_improvement_column = (
                columns.TestSetImprovementProbabilitySimple(
                    model,
                    dataset,
                    smoother=optimisers.smooth.MovingAverage(1, n=10),
                )
            )
            result.add_column(test_set_improvement_column)
        
        if args.dynamic_terminator:
            dynamic_terminator = optimisers.DynamicTerminator(
                model=model,
                dataset=dataset,
                batch_size=args.batch_size,
                replace=False,
                smooth_n=args.dt_buffer_length,
                t_lim=args.t_lim,
            )
            terminator = dynamic_terminator
            batch_getter = dynamic_terminator

            dynamic_terminator_column = columns.BatchImprovementProbability(
                dynamic_terminator,
            )
            result.add_column(dynamic_terminator_column)
        else:
            terminator = optimisers.Terminator(t_lim=args.t_lim)
            batch_getter = optimisers.batch.ConstantBatchSize(
                args.batch_size,
                False,
            )

        # Perform gradient descent
        optimisers.gradient_descent(
            model,
            dataset,
            line_search=args.line_search,
            result=result,
            evaluator=optimisers.Evaluator(t_interval=args.t_eval),
            terminator=terminator,
            batch_getter=batch_getter,
        )

        # Store results
        result_list.append(result)
        model_list.append(model)
        if args.plot_pred_gif or args.plot_hidden_gif:
            prediction_column_list.append(pred_column)
    
    # Make output plots
    print("Plotting output plots in \"%s\"..." % output_dir)
    os.system("explorer \"%s\"" % output_dir)
    print("Plotting training curves...")
    plotting.plot_training_curves(
        result_list,
        dir_name=output_dir,
        e_lims=args.error_lims,
    )
    if args.plot_test_set_improvement_probability or args.dynamic_terminator:
        attribute_list=[
            columns.TrainError,
            columns.TestError,
            columns.StepSize,
        ]
        if args.plot_test_set_improvement_probability:
            print("Plotting test set improvement probability...")
            attribute_list.append(columns.TestSetImprovementProbabilitySimple)
        if args.dynamic_terminator:
            print("Plotting batch improvement probability...")
            attribute_list.append(columns.BatchImprovementProbability)
        plotting.plot_result_attributes_subplots(
            plot_name="Improvement probability\n%s" % param_str,
            dir_name=output_dir,
            result_list=result_list,
            attribute_list=attribute_list,
            log_axes_attributes=[columns.StepSize],
            iqr_axis_scaling=True,
        )

    for i, model in enumerate(model_list):
        output_dir_repeat = os.path.join(output_dir, "Repeat %i" % (i + 1))
        if args.plot_preds:
            print("Plotting final predictions...")
            plotting.plot_data_predictions(
                plot_name="Final predictions",
                dir_name=output_dir_repeat,
                dataset=dataset,
                output_dim=args.output_dim,
                model=model,
            )
        if args.plot_pred_gif:
            print("Plotting gif of predictions during training...")
            plotting.plot_predictions_gif(
                plot_name="Model predictions during training",
                dir_name=output_dir_repeat,
                result=result_list[i],
                prediction_column=prediction_column_list[i],
                dataset=dataset,
                output_dim=args.output_dim,
                duration=args.t_eval*1000,
            )
        if args.plot_hidden_gif:
            print("Plotting gif of hidden layers during training...")
            if args.plot_hidden_preactivations_gif:
                plot_name="Hidden layer preactivations during training"
            else:
                plot_name="Hidden layer outputs during training"

            plotting.plot_hidden_outputs_gif(
                plot_name=plot_name,
                dir_name=output_dir_repeat,
                result=result_list[i],
                prediction_column=prediction_column_list[i],
                dataset=dataset,
                output_dim=args.output_dim,
                duration=args.t_eval*1000,
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
        dest="num_hidden_units_str",
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
    parser.add_argument(
        "--plot_test_set_improvement_probability",
        help="If this flag is included, then plot the probability of the test "
        "set error improving each time the Result object is updated",
        action="store_true",
    )
    parser.add_argument(
        "--dynamic_terminator",
        help="If this flag is included, then use a dynamic terminator, and "
        "plot the probability of improvement during each iteration for each "
        "repeat",
        action="store_true",
    )
    parser.add_argument(
        "--dt_buffer_length",
        help="Length of the buffer used by the dynamic terminator to smooth "
        "the estimated probability of improvement (a higher number means a "
        "smoother estimation and a slower reaction time)",
        type=int,
        default=100,
    )

    # Parse arguments
    args = parser.parse_args()

    args.num_hidden_units = [
        int(i) for i in args.num_hidden_units_str.split(",")
    ]
    
    args.line_search = None if args.no_line_search else optimisers.LineSearch()

    args.t_eval = args.t_lim / 50 if args.t_eval is None else args.t_eval

    args.dataset_type = data.dataset_class_dict[args.dataset_type_str]

    if args.error_lims is not None:
        float_fmt = lambda e: -float(e[1:]) if e.startswith("n") else float(e)
        args.error_lims = [float_fmt(e) for e in args.error_lims.split(",")]
        error_msg = "Must provide 2 comma-separated values for error_lims"
        assert len(args.error_lims) == 2, error_msg

    # Call main function using command-line arguments
    t_start = perf_counter()
    main(args)
    print("Main function run in %.3f s" % (perf_counter() - t_start))
