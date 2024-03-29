"""
Module containing unit tests for the Result class in the results module.

TODO: test exceptions in the Result class, EG adding a column which already
exists with the same name, calling Result.update before Result.begin, and
calling Result.display_last before calling Result.update
"""
import os
import pytest
import numpy as np
from optimisers import results, LineSearch
from .util import get_random_network, get_output_dir
import data, optimisers

# Get name of output directory, and create it if it doesn't exist
output_dir = get_output_dir("Result")

def get_updated_or_empty_result(use_updated_result, seed):
    if use_updated_result:
        return test_update(seed)
    else:
        result = results.Result()
        result.begin()
        return result

@pytest.mark.parametrize("seed", [3681, 7269, 2084])
def test_update(seed, result=None):
    """
    Test the update method of the Result class. Also return the result, which
    can be used as an input to other test functions. Can accept a pre-existing
    Result object, EG if it has been initialised with a specific output file
    """
    np.random.seed(seed)
    output_dim = np.random.randint(2, 5)
    n = get_random_network(input_dim=2, output_dim=output_dim)
    d = data.Sinusoidal(input_dim=2, output_dim=output_dim, n_train=100)
    w = n.get_parameter_vector()
    if result is None:
        result = results.Result()

    result.begin()
    # Call the result.update method a few times
    for i in range(5):
        n.set_parameter_vector(w + i)
        result.update(model=n, dataset=d, iteration=i)

    # Check that value lists for the default columns have non-zero length
    assert len(result.get_values(optimisers.results.columns.TrainError)) > 0
    assert len(result.get_values(optimisers.results.columns.TestError)) > 0
    assert len(result.get_values(optimisers.results.columns.Iteration)) > 0
    assert len(result.get_values(optimisers.results.columns.Time)) > 0

    return result

@pytest.mark.parametrize("seed", [9843, 1213, 1005])
@pytest.mark.parametrize("use_updated_result", [True, False])
def test_display_headers(seed, use_updated_result):
    """ Test the display_headers method of the Result class """
    result = get_updated_or_empty_result(use_updated_result, seed)

    result.display_headers()

@pytest.mark.parametrize("seed", [9190, 6940, 6310])
@pytest.mark.parametrize("use_updated_result", [True])
def test_display_last(seed, use_updated_result):
    """ Test the display_last method of the Result class """
    result = get_updated_or_empty_result(use_updated_result, seed)

    result._display_last()

@pytest.mark.parametrize("seed", [4973, 6153, 4848])
@pytest.mark.parametrize("use_updated_result", [True, False])
def test_display_summary(seed, use_updated_result):
    """ Test the display_summary method of the Result class """
    result = get_updated_or_empty_result(use_updated_result, seed)

    result.display_summary(10)

def test_output_file():
    """
    Test the methods of a Result object, writing the outputs to a text file
    """
    seed = 8151
    np.random.seed(seed)
    filename = "Result tests output.txt"
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        result = results.Result("Test result", True, f)
        test_update(seed=seed, result=result)
        result.display_headers()
        result._display_last()
        result.display_summary(10)

@pytest.mark.parametrize("seed", [6953, 485, 5699])
def test_line_search_column(seed):
    """
    Test that changes in the step size attribute of a LineSearch object are
    reflected in the value lists of a Result object when the Result object is
    updated.
    """
    np.random.seed(seed)
    num_steps = np.random.randint(10, 20)
    step_sizes = np.random.uniform(0, 10, num_steps)
    filename = "Test StepSize results column.txt"
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        result = results.Result(add_default_columns=False, file=f)
        ls = LineSearch()
        col = results.columns.StepSize(ls)
        result.add_column(col)
        result.begin()
        for s in step_sizes:
            ls.s = s
            result.update()

        assert np.all(result.get_values(type(col)) == step_sizes)

def test_get_iteration_number():
    """ Unit test for the public get_iteration_number method of the Result class
    """
    # Create result object, and tell it to begin
    result = results.Result(add_default_columns=False)
    result.begin()
    # Check the initial iteration number is 0
    assert result.get_iteration_number() == 0
    # Check that the iteration number is updated when calling update
    for i in range(10):
        result.update(iteration=i)
        assert result.get_iteration_number() == i
