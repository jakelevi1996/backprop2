""" Module containing unit tests for columns objects, used to verify that the
columns are funtioning correctly, and also to serve as usage examples for each
column. """
import os
from math import ceil
import numpy as np
import pytest
import models, data, optimisers
from .util import set_random_seed_from_args, get_random_network, get_output_dir

# Get name of output directory, and create it if it doesn't exist
output_dir = get_output_dir("Columns")

def test_standard_columns():
    """ Test using a Result object with the standard columns, which are added
    to a Result object by default """
    # Initialise random seed and number of training data points and iterations
    set_random_seed_from_args("test_standard_columns")
    n_train = np.random.randint(10, 20)
    n_its = np.random.randint(10, 20)
    # Initialise model and data set
    model = get_random_network(input_dim=1, output_dim=1)
    sin_data = data.Sinusoidal(input_dim=1, output_dim=1, n_train=n_train)
    # Initialise output file
    test_name = "Test standard columns"
    output_filename = "test_standard_columns.txt"
    with open(os.path.join(output_dir, output_filename), "w") as f:
        # Initialise result object
        result = optimisers.Result(
            name=test_name,
            file=f,
            add_default_columns=True
        )
        # Perform optimisation
        optimisers.gradient_descent(
            model,
            sin_data,
            result=result,
            terminator=optimisers.Terminator(i_lim=n_its),
            evaluator=optimisers.Evaluator(i_interval=1)
        )
    # Check that each column object has the correct number of values
    for col_type in optimisers.results.DEFAULT_COLUMN_TYPES:
        # "n_its + 1" because we evaluate the initial state of the model
        assert len(result.get_values(col_type)) == n_its + 1

def test_step_size_column():
    """ Test using step size column with a Result object """
    set_random_seed_from_args("test_step_size_column")
    n_train = np.random.randint(10, 20)
    n_its = np.random.randint(10, 20)
    model = get_random_network(input_dim=1, output_dim=1)
    sin_data = data.Sinusoidal(input_dim=1, output_dim=1, n_train=n_train)
    test_name = "Test line search column"
    output_filename = "test_step_size_column.txt"
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
    set_random_seed_from_args("test_dbs_column")
    n_train = np.random.randint(10, 20)
    n_its = np.random.randint(10, 20)
    model = get_random_network(input_dim=1, output_dim=1)
    sin_data = data.Sinusoidal(input_dim=1, output_dim=1, n_train=n_train)
    # Initialise output file and Result object
    test_name = "Test DBS column"
    output_filename = "test_dbs_columns.txt"
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
    set_random_seed_from_args("test_optimal_batch_size_column")
    n_train = np.random.randint(10, 20)
    n_its = np.random.randint(10, 20)
    model = get_random_network(input_dim=1, output_dim=1)
    sin_data = data.Sinusoidal(input_dim=1, output_dim=1, n_train=n_train)
    # Initialise output file and Result object
    test_name = "Test optimal batch size column"
    output_filename = "test_optimal_batch_size_column.txt"
    with open(os.path.join(output_dir, output_filename), "w") as f:
        result = optimisers.Result(
            name=test_name,
            file=f,
            add_default_columns=True
        )
        # Initialise line-search and column object, and add to the result
        n_batch_sizes = np.random.randint(3, 6)
        n_repeats = np.random.randint(3, 6)
        line_search = optimisers.LineSearch()
        columns = optimisers.results.columns
        gd_optimiser = optimisers.GradientDescent(line_search)
        optimal_batch_size_col = columns.OptimalBatchSize(
            gd_optimiser,
            sin_data.n_train,
            n_repeats=n_repeats,
            n_batch_sizes=n_batch_sizes
        )
        result.add_column(optimal_batch_size_col)
        # Call optimisation function
        gd_optimiser.optimise(
            model,
            sin_data,
            result=result,
            terminator=optimisers.Terminator(i_lim=n_its),
            evaluator=optimisers.Evaluator(i_interval=1),
        )
    # Test that the OptimalBatchSize object attributes are as expected
    batch_size_list = optimal_batch_size_col.batch_size_list
    assert len(optimal_batch_size_col.reduction_dict_dict) == (n_its + 1)
    for reduction_dict in optimal_batch_size_col.reduction_dict_dict.values():
        assert len(reduction_dict) == n_batch_sizes
        assert set(reduction_dict.keys()) == set(batch_size_list)
        for reduction_list in reduction_dict.values():
            assert len(reduction_list) == n_repeats

@pytest.mark.parametrize("store_hidden", [False, True])
@pytest.mark.parametrize("store_preactivations", [False, True])
@pytest.mark.parametrize("input_dim, output_dim", [(1, 1), (2, 3)])
def test_predictions_column(
    input_dim,
    output_dim,
    store_hidden,
    store_preactivations,
):
    """ Test using a column which stores model predictions during training """
    # Set random seed and initialise network and dataset
    set_random_seed_from_args(
        "test_predictions_column",
        input_dim,
        output_dim,
        store_hidden,
    )
    n_train = np.random.randint(10, 20)
    n_pred = ceil(pow(np.random.randint(5, 10), 1/input_dim))
    n_its = np.random.randint(10, 20)
    model = get_random_network(input_dim=input_dim, output_dim=output_dim)
    sin_data = data.Sinusoidal(
        input_dim=input_dim,
        output_dim=output_dim,
        n_train=n_train,
    )
    # Initialise output file and Result object
    test_name = "test_predictions_column, %id-%id data, store_hidden=%s" % (
        input_dim,
        output_dim,
        store_hidden,
    )
    output_filename = "%s.txt" % test_name
    with open(os.path.join(output_dir, output_filename), "w") as f:
        # Initialise result object
        result = optimisers.Result(
            name=test_name,
            file=f,
            add_default_columns=True
        )
        # Initialise column object and add to the result
        columns = optimisers.results.columns
        prediction_column = columns.Predictions(
            sin_data,
            n_points_per_dim=n_pred,
            store_hidden_layer_outputs=store_hidden,
            store_hidden_layer_preactivations=store_preactivations,
        )
        result.add_column(prediction_column)
        # Call optimisation function
        optimisers.gradient_descent(
            model,
            sin_data,
            result=result,
            terminator=optimisers.Terminator(i_lim=n_its),
            evaluator=optimisers.Evaluator(i_interval=1),
        )
        # Print Predictions column attributes to file
        print("\n\nx_pred:", prediction_column.x_pred, sep="\n", file=f)
        iter_list = result.get_values(columns.Iteration)
        print("\n\nPredictions:", file=f)
        for i in iter_list:
            print("i = %i:" % i, file=f)
            print(prediction_column.predictions_dict[i], file=f)
        if store_hidden:
            print("\n\nHidden layer outputs:", file=f)
            for i in iter_list:
                print(
                    "\ni = %i:" % i,
                    *prediction_column.hidden_outputs_dict[i],
                    file=f,
                    sep="\n\n",
                )

    # Test that the Prediction object attributes are as expected
    n_pred_grid = pow(n_pred, input_dim)
    assert prediction_column.x_pred.shape == (input_dim, n_pred_grid)
    
    iter_set = set(iter_list)
    assert set(prediction_column.predictions_dict.keys()) == iter_set
    for y_pred in prediction_column.predictions_dict.values():
        assert y_pred.shape == (output_dim, n_pred_grid)
    
    hidden_outputs_dict = prediction_column.hidden_outputs_dict
    if store_hidden:
        assert set(hidden_outputs_dict.keys()) == iter_set
        for hidden_output_list in hidden_outputs_dict.values():
            assert len(hidden_output_list) == len(model.layers) - 1
            for i, hidden_output in enumerate(hidden_output_list):
                expected_shape = (model.layers[i].output_dim, n_pred_grid)
                assert hidden_output.shape == expected_shape
    else:
        assert len(hidden_outputs_dict) == 0
