"""
Module containing unit tests for the Result class in the results module.
"""
import os
import pytest
import numpy as np
from optimisers import results, LineSearch
from .util import get_random_network
import data


# Get name of output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs")

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
    d = data.SinusoidalDataSet2DnD(nx0=10, nx1=15, output_dim=output_dim)
    w = n.get_parameter_vector()
    if result is None:
        result = results.Result()
    
    result.begin()
    # Call the result.update method a few times
    for i in range(5):
        n.set_parameter_vector(w + i)
        result.update(model=n, dataset=d, iteration=i)
    
    # Check that value lists for the default columns have non-zero length
    assert len(result.get_values("train_error")) > 0
    assert len(result.get_values("test_error")) > 0
    assert len(result.get_values("iteration")) > 0
    assert len(result.get_values("time")) > 0
    
    return result

@pytest.mark.parametrize("seed", [810, 6361, 9133])
@pytest.mark.parametrize("use_updated_result", [True, False])
def test_save_load(seed, use_updated_result):
    """ Test saving and loading a Result object """
    result = get_updated_or_empty_result(use_updated_result, seed)

    filename = "Saved result.npz"
    result.save(filename, output_dir)
    loaded_result = results.load(filename, output_dir)
    
    assert result.name          == loaded_result.name
    assert result.verbose       == loaded_result.verbose
    assert result.train_errors  == loaded_result.train_errors
    assert result.train_errors  == loaded_result.train_errors
    assert result.test_errors   == loaded_result.test_errors
    assert result.times         == loaded_result.times
    assert result.iters         == loaded_result.iters
    assert result.step_size     == loaded_result.step_size
    assert result.start_time    == loaded_result.start_time

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

    result.display_last()

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
        result.display_last()
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
        result = results.Result(add_default_columns=False)
        ls = LineSearch()
        col = results.columns.StepSize(ls)
        result.add_column(col)
        result.begin()
        for s in step_sizes:
            ls.s = s
            result.update()
        
        assert np.all(result.get_values(col.name) == step_sizes)
