import os
import numpy as np
import pytest
import data
from .util import dataset_dict, get_output_dir, set_random_seed_from_args

# Get name of output directory, and create it if it doesn't exist
output_dir = get_output_dir("Data")

@pytest.mark.parametrize("dataset_key", dataset_dict.keys())
def test_save_load(dataset_key):
    """
    Test initialising a dataset subclass with default constructor arguments,
    check that it can be saved and loaded, and that the saved and loaded dataset
    objects are equivalent
    """
    set_random_seed_from_args("test_save_load", dataset_key)
    dataset = dataset_dict[dataset_key]
    dataset.save(dataset_key, output_dir)
    dataset_loaded = data.DataSet(dataset_key, output_dir)
    attr_list = [
        "input_dim" , "output_dim"  ,
        "n_train"   , "n_test"      ,
        "x_train"   , "x_test"      ,
        "y_train"   , "y_test"
    ]
    for a in attr_list:
        assert np.all(getattr(dataset, a) == getattr(dataset_loaded, a))

@pytest.mark.parametrize("dataset_key", dataset_dict.keys())
def test_print_data(dataset_key):
    set_random_seed_from_args("test_print_data", dataset_key)
    dataset = dataset_dict[dataset_key]
    # Print data to stdout
    dataset.print_data()
    # Print data to file
    filename = dataset_key + " data.txt"
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
