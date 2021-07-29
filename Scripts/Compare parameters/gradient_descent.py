"""
Find best parameters for gradient descent with line-search, by trying different
combinations and plotting the results. The different parameters that are
compared are:
-   Number of units
-   Number of layers
-   s0 (initial step size)
-   alpha (threshold for backtracking)
-   beta (ratio of changes in step size)
-   The maximum number of steps taken during each iteration of line search
-   Activation function

Results are compared by plotting final performance after a fixed length of time
allowed for optimisation.

This script has a command-line interface. Below are some examples for calling
this script:

    python "Scripts\Compare parameters\gradient_descent.py"

    python "Scripts\Compare parameters\gradient_descent.py" -i2 -o3 -n2500

    python "Scripts\Compare parameters\gradient_descent.py" -t"0.1"

    python "Scripts\Compare parameters\gradient_descent.py" -f

To get help information for the available arguments, use the following command:

    python "Scripts\Compare parameters\gradient_descent.py" -h
"""
import os
import argparse
from time import perf_counter
import numpy as np
if __name__ == "__main__":
    import __init__
import models
import data
import optimisers
import plotting
from experiment import Experiment, Parameter

def main(args):
    # Get name of output directory, and create it if it doesn't exist
    param_str = (
        "input_dim = %i, output_dim = %i, n_train = %i, t_lim = %.2f, "
        "num_repeats = %i, find_best_params = %s"
        % (
            args.input_dim,
            args.output_dim,
            args.n_train,
            args.t_lim,
            args.n_repeats,
            args.find_best_params,
        )
    )
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(
        current_dir,
        "Outputs",
        "Gradient descent",
        param_str,
    )
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Initialise data set
    np.random.seed(6763)
    sin_data = data.Sinusoidal(
        args.input_dim,
        args.output_dim,
        args.n_train,
        x_lo=-2,
        x_hi=2,
        freq=(1 if args.input_dim == 1 else None),
    )

    # Define function to be run for each experiment
    def run_experiment(
        num_units,
        num_layers,
        log10_s0,
        alpha,
        beta,
        act_func,
        max_steps,
        batch_size,
        batch_replace,
    ):
        # Initialise network and batch getter
        model = models.NeuralNetwork(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            num_hidden_units=[num_units for _ in range(num_layers)],
            act_funcs=[act_func, models.activations.identity],
        )

        if (batch_size is None) or (batch_size >= args.n_train):
            batch_getter = optimisers.batch.FullTrainingSet()
        else:
            batch_getter = optimisers.batch.ConstantBatchSize(
                batch_size,
                batch_replace,
            )
        
        # Perform gradient descent
        result = optimisers.gradient_descent(
            model,
            sin_data,
            terminator=optimisers.Terminator(t_lim=args.t_lim),
            evaluator=optimisers.Evaluator(t_interval=args.t_eval),
            line_search=optimisers.LineSearch(
                s0=pow(10, log10_s0), 
                alpha=alpha, 
                beta=beta,
                max_its=max_steps,
            ),
            batch_getter=batch_getter,
        )
        # Return the final test error
        TestError = optimisers.results.columns.TestError
        final_test_error = result.get_values(TestError)[-1]
        return final_test_error

    # Initialise the Experiment object, and add parameters
    experiment = Experiment(run_experiment, output_dir, args.n_repeats)
    addp = lambda *args: experiment.add_parameter(Parameter(*args))
    addp("num_units",       10,     [5, 10, 15, 20]                         )
    addp("num_layers",      1,      [1, 2, 3]                               )
    addp("log10_s0",        0,      np.linspace(-1, 3, 5)                   )
    addp("alpha",           0.5,    np.linspace(0.5, 1, 5, endpoint=False)  )
    addp("beta",            0.5,    np.linspace(0.5, 1, 5, endpoint=False)  )
    addp("max_steps",       10,     [5, 10, 15, 20]                         )
    addp("batch_size",      100,    [25, 50, 75, 100, 150, 200, 300, None]  )
    addp("batch_replace",   True,   [True, False]                           )
    addp(
        "act_func",
        models.activations.gaussian,
        [
            models.activations.gaussian,
            models.activations.cauchy,
            models.activations.logistic,
            models.activations.relu,
        ],
    )

    # Call warmup function
    optimisers.warmup()

    # Call function to run all experiments
    if args.find_best_params:
        experiment.find_best_parameters()
    else:
        experiment.sweep_all_parameters()
    
    # Write the results of all experiments to a text file
    experiment.save_results_as_text()

    # Open the output plot directory
    os.system("explorer \"%s\"" % output_dir)

if __name__ == "__main__":
    # Define CLI using argparse
    parser = argparse.ArgumentParser(
        description="Compare parameters for gradient descent with a line "
        "search"
    )

    parser.add_argument(
        "-i",
        "--input_dim",
        help="Number of input dimensions for the data set",
        default=1,
        type=int,
    )
    parser.add_argument(
        "-o",
        "--output_dim",
        help="Number of output dimensions for the data set",
        default=1,
        type=int,
    )
    parser.add_argument(
        "-n",
        "--n_train",
        help="Number of data points in the training set",
        default=50,
        type=int,
    )
    parser.add_argument(
        "-t",
        "--t_lim",
        help="How long to run each experiment for in seconds",
        default=3,
        type=float,
    )
    parser.add_argument(
        "-e",
        "--t_eval",
        help="How frequently to evaluate the performance of each model in "
        "seconds",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "-r",
        "--n_repeats",
        help="Number of repeats to perform of each experiment",
        default=5,
        type=int,
    )
    parser.add_argument(
        "-f",
        "--find_best_params",
        help="Iterate experiments and update parameters until the best "
        "parameters have been found",
        action="store_true",
    )

    # Parse arguments
    args = parser.parse_args()

    # Call main function using command-line arguments
    t_start = perf_counter()
    main(args)

    # Print time taken
    t_total = perf_counter() - t_start
    mins, secs = divmod(t_total, 60)
    print("\n\nScript ran in %i mins, %.3f secs" % (mins, secs))
