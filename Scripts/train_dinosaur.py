""" Train a Dinosaur model for meta-learning on a simple synthetic task-set,
and plot the final predictions, for tasks in the training task-set, as well as
a task outside of the training task-set, both with and without regularisation,
as well as the learning curves.

Below are some examples for calling this script:

    python Scripts/train_dinosaur.py --regulariser Quadratic

    python Scripts/train_dinosaur.py --regulariser Quartic

    python Scripts/train_dinosaur.py --regulariser QuarticType2

    python Scripts/train_dinosaur.py --regulariser QuarticType2 --error_scale_coefficient 1e-1

    python Scripts/train_dinosaur.py --regulariser QuarticType2 --num_hidden_units 20,20 --use_mnist_data

    python Scripts/train_dinosaur.py --regulariser QuarticType2 --num_hidden_units 20,20 --use_mnist_data --error_scale_coefficient 1e-1

    python Scripts/train_dinosaur.py --regulariser QuarticType3

To get help information for the available arguments, use the following command:

    python Scripts/train_dinosaur.py -h

"""

import os
import shutil
import argparse
import time
import numpy as np
if __name__ == "__main__":
    import __init__
import models
import data
import optimisers
import plotting

# Define input and output dimensions for models and data
INPUT_DIM = 2
OUTPUT_DIM = 1

def main(args):
    np.random.seed(args.seed)

    # Initialise network model
    network = models.NeuralNetwork(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        num_hidden_units=args.num_hidden_units,
    )

    # Get output directory which is specific to the relevant script parameters
    param_str = " ".join([
        "mnist" if args.use_mnist_data else "synthetic",
        "r%s"   % args.regulariser,
        "e%s"   % args.error_scale_coefficient,
        "u%s"   % args.num_hidden_units,
    ])
    if args.use_mnist_data:
        param_str += " " + " ".join([
            "t%s"   % args.mnist_num_train_tasks,
            "l%s"   % args.mnist_train_distribution_label,
            "o%s"   % args.mnist_out_of_distribution_label,
        ])
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(
        current_dir,
        "Outputs",
        "Train Dinosaur",
        param_str,
    )
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print("Saving output plots in \"%s\"" % output_dir)
    os.system("explorer \"%s\"" % output_dir)

    # Initialise data
    if args.use_mnist_data:
        task_set, out_of_distribution_task = get_mnist_data(args)
    else:
        task_set, out_of_distribution_task = get_synthetic_data()

    # Initialise meta-learning model
    regulariser_type = models.dinosaur.regularisers.regulariser_names_dict[
        args.regulariser
    ]
    regulariser = regulariser_type(
        error_scale_coefficient=args.error_scale_coefficient,
    )
    dinosaur = models.Dinosaur(
        network=network,
        regulariser=regulariser,
        primary_initialisation_task=task_set.task_list[0],
        secondary_initialisation_task=task_set.task_list[1],
    )

    for _ in range(10):
        # Perform one outer-loop iteration of meta-learning
        dinosaur._result.display_headers()
        dinosaur.meta_learn(
            task_set,
            terminator=optimisers.Terminator(i_lim=1),
        )
        # Check that the mean and scale are converging to sensible values
        print(regulariser.mean)
        print(regulariser.parameter_scale)
        print(regulariser.error_scale)
        # Compare adapting to an out-of-distribution task
        dinosaur.fast_adapt(out_of_distribution_task)

    # Plot training curves
    plotting.plot_training_curves([dinosaur._result], dir_name=output_dir)

    # Plot task predictions after meta-learning
    for i, task in enumerate(task_set.task_list):
        print("Plotting adaptations to task %i" % i)
        dinosaur.fast_adapt(task)
        plotting.plot_2D_regression(
            "Dinosaur task %i" % i,
            output_dir,
            task,
            OUTPUT_DIM,
            model=network,
        )

    # Plot adaptation to out of distribution task
    print("Plotting adaptation to out of distribution task")
    dinosaur.fast_adapt(out_of_distribution_task)
    plotting.plot_2D_regression(
        "Dinosaur predictions for out-of-distribution task",
        output_dir,
        out_of_distribution_task,
        OUTPUT_DIM,
        model=network,
    )

    # Plot adaptation to out of distribution task without regularisation
    print("Plotting adaptation without regularisation")
    network._regulariser.error_scale = 0
    network.set_parameter_vector(regulariser.mean)
    dinosaur.fast_adapt(out_of_distribution_task)
    plotting.plot_2D_regression(
        "Dinosaur predictions for out-of-distribution task without "
        "regularisation",
        output_dir,
        out_of_distribution_task,
        OUTPUT_DIM,
        model=network,
    )


def get_synthetic_data():
    # Initialise task set
    task_set = data.TaskSet()
    create_task = lambda x, y: data.GaussianCurve(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        n_train=2000,
        n_test=2000,
        x_lo=-1,
        x_hi=1,
        input_offset=np.array([x, y]).reshape([INPUT_DIM, 1]),
        input_scale=5,
        output_offset=0,
        output_scale=10,
    )
    for x in [0.3, 0.9]:
        for y in [0.5, 0.7]:
            # Initialise task
            task = create_task(x, y)
            # Add task to task set
            task_set.add_task(task)

    # Create out-of-distribution task
    out_of_distribution_task = create_task(-0.5, -0.5)

    return task_set, out_of_distribution_task

def get_mnist_data(args):
    task_set = data.TaskSet()
    mnist_task_map = data.Mnist()
    task_batch = mnist_task_map.train.get_batch(
        args.mnist_train_distribution_label,
        args.mnist_num_train_tasks,
    )
    for task in task_batch:
        task_set.add_task(task)
    out_of_distribution_task = mnist_task_map.train.get_batch(
        args.mnist_out_of_distribution_label,
        1,
    )[0]
    return task_set, out_of_distribution_task


if __name__ == "__main__":

    # Define CLI using argparse
    parser = argparse.ArgumentParser(
        description="Train a Dinosaur model for meta-learning on a simple "
        "synthetic task-set"
    )

    parser.add_argument(
        "--seed",
        help="Seed to use for random number generation, should be a positive "
        "integer",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--regulariser",
        help="Which regulariser to use with the meta-learning model",
        default="Quadratic",
        type=str,
        choices=models.dinosaur.regularisers.regulariser_names_dict.keys(),
    )
    parser.add_argument(
        "--error_scale_coefficient",
        help="Constant coefficient to multiply by the error scale when "
        "updating the regulariser parameters",
        default=1e-2,
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
        "--use_mnist_data",
        help="Use Mnist data instead of synthetic Gaussian curve data",
        action="store_true",
    )
    parser.add_argument(
        "--mnist_num_train_tasks",
        help="Number of training tasks to use if using Mnist data",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--mnist_train_distribution_label",
        help="Digit label to train on if using Mnist data",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--mnist_out_of_distribution_label",
        help="Digit label to compare adaptation with training distribution if "
        "using Mnist data",
        type=int,
        default=2,
    )

    # Parse arguments
    args = parser.parse_args()

    args.num_hidden_units = [
        int(i) for i in args.num_hidden_units_str.split(",")
    ]

    # Call main function using command-line arguments
    t_start = time.perf_counter()
    main(args)
    print("Main function run in %.3f s" % (time.perf_counter() - t_start))
