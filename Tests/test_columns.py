""" Unit test module for columns objects, with one test for each different
column, used to verify that the columns are funtioning correctly, and also to
serve as usage examples for each column. TODO """
import os
import numpy as np
import pytest
import models, data, optimisers
from .util import get_random_network
from .util import output_dir as parent_output_dir

# Get name of output directory, and create it if it doesn't exist
output_dir = os.path.join(parent_output_dir, "Test columns")
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

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
    np.random.seed(1522)
    n_train = np.random.randint(10, 20)
    n_its = np.random.randint(10, 20)
    model = get_random_network(input_dim=1, output_dim=1)
    sin_data = data.Sinusoidal(input_dim=1, output_dim=1, n_train=n_train)
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
