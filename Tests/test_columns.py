""" Unit test module for columns objects, with one test for each different
column, used to verify that the columns are funtioning correctly, and also to
serve as usage examples for each column. TODO """
import os
import numpy as np
import pytest
import models, data, optimisers
from .util import get_random_network, get_output_dir

# Get name of output directory, and create it if it doesn't exist
output_dir = get_output_dir("Columns")

def test_standard_columns():
    """ Test using a Result object with the standard columns """
    np.random.seed(1449)
    n_train = np.random.randint(10, 20)
    n_its = np.random.randint(10, 20)
    model = get_random_network(input_dim=1, output_dim=1)
    sin_data = data.Sinusoidal(input_dim=1, output_dim=1, n_train=n_train)
    test_name = "Test standard columns"
    output_filename = "%s.txt" % test_name
    with open(os.path.join(output_dir, output_filename), "w") as f:
        result = optimisers.Result(
            name=test_name,
            file=f,
            add_default_columns=True
        )
        optimisers.gradient_descent(
            model,
            sin_data,
            result=result,
            terminator=optimisers.Terminator(i_lim=n_its),
            evaluator=optimisers.Evaluator(i_interval=1)
        )

def test_step_size_column():
    """ Test using step size column with a Result object """
    np.random.seed(1522)
    n_train = np.random.randint(10, 20)
    n_its = np.random.randint(10, 20)
    model = get_random_network(input_dim=1, output_dim=1)
    sin_data = data.Sinusoidal(input_dim=1, output_dim=1, n_train=n_train)
    test_name = "Test line search column"
    output_filename = "%s.txt" % test_name
    with open(os.path.join(output_dir, output_filename), "w") as f:
        ls = optimisers.LineSearch()
        result = optimisers.Result(
            name=test_name,
            file=f,
            add_default_columns=True
        )
        result.add_column(optimisers.results.columns.StepSize(ls))
        optimisers.gradient_descent(
            model,
            sin_data,
            result=result,
            terminator=optimisers.Terminator(i_lim=n_its),
            evaluator=optimisers.Evaluator(i_interval=1),
            line_search=ls
        )

def test_dbs_column():
    """ Test using a DBS column with a Result object """
    # Set random seed and initialise network and dataset
    np.random.seed(1522)
    n_train = np.random.randint(10, 20)
    n_its = np.random.randint(10, 20)
    model = get_random_network(input_dim=1, output_dim=1)
    sin_data = data.Sinusoidal(input_dim=1, output_dim=1, n_train=n_train)
    # Initialise output file and Result object
    test_name = "Test DBS column"
    output_filename = "%s.txt" % test_name
    with open(os.path.join(output_dir, output_filename), "w") as f:
        result = optimisers.Result(
            name=test_name,
            file=f,
            add_default_columns=True
        )
        result.add_column(optimisers.results.columns.DbsMetric())
        # Initialise gradient vector before DBS is calculated
        model.get_gradient_vector(sin_data.x_train, sin_data.x_test)
        optimisers.gradient_descent(
            model,
            sin_data,
            result=result,
            terminator=optimisers.Terminator(i_lim=n_its),
            evaluator=optimisers.Evaluator(i_interval=1)
        )

def test_optimal_batch_size_column():
    """ Test using a column which approximates the optimal batch size on each
    iteration """
    # Set random seed and initialise network and dataset
    np.random.seed(2231)
    n_train = np.random.randint(10, 20)
    n_its = np.random.randint(10, 20)
    n_repeats = np.random.randint(5, 10)
    n_batch_sizes = np.random.randint(5, 10)
    model = get_random_network(input_dim=1, output_dim=1)
    sin_data = data.Sinusoidal(input_dim=1, output_dim=1, n_train=n_train)
    # Initialise output file and Result object
    test_name = "Test optimal batch size column"
    output_filename = "%s.txt" % test_name
    with open(os.path.join(output_dir, output_filename), "w") as f:
        result = optimisers.Result(
            name=test_name,
            file=f,
            add_default_columns=True
        )
        # Initialise line-search and column object, and add to the result
        line_search = optimisers.LineSearch()
        columns = optimisers.results.columns
        optimal_batch_size_col = columns.OptimalBatchSize(
            model,
            sin_data,
            line_search,
            optimisers.gradient_descent,
            n_repeats=3,
            n_batch_sizes=4
        )
        result.add_column(optimal_batch_size_col)
        # Call optimisation function
        optimisers.gradient_descent(
            model,
            sin_data,
            result=result,
            line_search=line_search,
            terminator=optimisers.Terminator(i_lim=n_its),
            evaluator=optimisers.Evaluator(i_interval=1),
        )

