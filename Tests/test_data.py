import os
import numpy as np
import pytest
import data
from .util import get_dataset, dataset_list, get_output_dir

# Get name of output directory, and create it if it doesn't exist
output_dir = get_output_dir("Data")

@pytest.mark.parametrize("seed", [4405, 9721, 5974])
@pytest.mark.parametrize("dataset_str", dataset_list)
def test_save_load(seed, dataset_str):
    """
    Test initialising a dataset subclass with default constructor arguments,
    check that it can be saved and loaded, and that the saved and loaded dataset
    objects are equivalent
    """
    np.random.seed(seed)
    dataset = get_dataset(dataset_str)
    dataset.save(dataset_str, output_dir)
    dataset_loaded = data.DataSet(dataset_str, output_dir)
    attr_list = [
        "input_dim" , "output_dim"  ,
        "n_train"   , "n_test"      ,
        "x_train"   , "x_test"      ,
        "y_train"   , "y_test"
    ]
    for a in attr_list:
        assert np.all(getattr(dataset, a) == getattr(dataset_loaded, a))

@pytest.mark.parametrize("seed", [1854, 7484, 5736])
@pytest.mark.parametrize("dataset_str", dataset_list)
def test_print_data(seed, dataset_str):
    np.random.seed(seed)
    dataset = get_dataset(dataset_str)
    # Print data to stdout
    dataset.print_data()
    # Print data to file
    filename = dataset_str + "data.txt"
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        dataset.print_data(file=f)

def test_valid_xlim():
    """ Test initialising a DataSet with valid single- and multi-dimensional
    x-limits """
    # Test single-dimensional x-limits
    d = data.Sinusoidal(input_dim=4, x_hi=7.5, n_train=10)

    # Test multi-dimensional x-limits
    x_hi = np.array([3, 4, 5, 6]).reshape(-1, 1)
    d = data.Sinusoidal(input_dim=4, x_hi=x_hi, n_train=10)

def test_invalid_xlim():
    """ Test that a ValueError is raised when initialising a DataSet with
    x-limits that don't broadcast to the size of x_train and x_test.

    As stated in the docstring for the data.Sinusoidal initialiser, x_hi "should
    be a float, or a numpy array with shape [input_dim, 1]". """
    with pytest.raises(ValueError):
        d = data.Sinusoidal(input_dim=3, x_lo=[1, 2])
    
    with pytest.raises(ValueError):
        d = data.Sinusoidal(input_dim=4, x_hi=[3, 4, 5, 6], n_train=10)
    
    with pytest.raises(ValueError):
        x_hi = np.array([3, 4, 5, 6]).reshape(1, -1)
        d = data.Sinusoidal(input_dim=4, x_hi=x_hi, n_train=10)

def test_invalid_freq_shape():
    """
    Test that initialising a DataSet with a frequency that doesn't broadcast to
    the input and output dimensions raises a ValueError
    """
    with pytest.raises(ValueError):
        d = data.Sinusoidal(input_dim=3, output_dim=7, freq=np.zeros([5, 9]))
