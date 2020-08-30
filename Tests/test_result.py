"""
Module containing unit tests for the Result class in the results module.
"""
import os
import pytest
import numpy as np
import results
from .util import get_random_network
import data


# Get name of output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs")

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
    
    # Call the result.update method a few times
    for i in range(5):
        s = np.random.normal()
        n.set_parameter_vector(w + i)
        result.update(n, d, i, s)
    
    # Check that some of the attributes have non-zero length
    assert len(result.train_errors) > 0
    assert len(result.times) > 0
    assert len(result.step_size) > 0
    
    return result

# Create an empty result, and get an updated result, for testing other methods
empty_result = results.Result()
updated_result = test_update(8946)

@pytest.mark.parametrize("result", [empty_result, updated_result])
def test_save_load(result):
    """ Test saving and loading a Result object """
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

@pytest.mark.parametrize("result", [empty_result, updated_result])
def test_display_headers(result):
    """ Test the display_headers method of the Result class """
    result.display_headers()

@pytest.mark.parametrize("result", [updated_result])
def test_display_last(result):
    """ Test the display_last method of the Result class """
    result.display_last()

@pytest.mark.parametrize("result", [empty_result, updated_result])
def test_display_summary(result):
    """ Test the display_summary method of the Result class """
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
