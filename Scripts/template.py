""" Template script, containing command-line arguments.

Below are some examples for calling this script:

    python Scripts/template.py -i1 -o1 -n100 -b50 -t1 -u10 -l0 -g 0.02 -r3

    python Scripts/template.py -i2 -o3 -n2500 -b50 -t10 -u 20,20 -l0 -g4 -r3

Running each of the above examples requires 12.858 s and 121.004 s respectively.

To get help information for the available arguments, use the following command:

    python Scripts/template.py -h

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
    e_lims,
    n_repeats
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
    -   n_repeats: positive integer number of repeats to perform of each
        experiment
    """
    np.random.seed(1913)

    # Perform warmup experiment so process acquires priority
    optimisers.warmup()

    # Do something useful
    pass
    

if __name__ == "__main__":
    
    # Define CLI using argparse
    parser = ArgumentParser(
        description="Compare generalised Newton's method vs gradient descent"
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
        help="Batch size to use for training",
        default=50,
        type=int
    )
    parser.add_argument(
        "-t",
        "--t_lim",
        help="Length of time to train for each experiment",
        default=5,
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
        "-l",
        "--e_lo",
        help="Lower axis limits for output plot",
        default=0,
        type=float
    )
    parser.add_argument(
        "-g",
        "--e_hi",
        help="Upper axis limits for output plot",
        default=0.2,
        type=float
    )
    parser.add_argument(
        "-r",
        "--n_repeats",
        help="Number of repeats to perform of each experiment",
        default=3,
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
        args.batch_size,
        args.t_lim,
        num_hidden_units,
        [args.e_lo, args.e_hi],
        args.n_repeats,
    )
    print("Main function run in %.3f s" % (perf_counter() - t_start))
