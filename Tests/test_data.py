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

def test_invalid_freq_shape():
    """
    Test that initialising a DataSet with a frequency that doesn't broadcast to
    the input and output dimensions raises a ValueError
    """
    with pytest.raises(ValueError):
        d = data.Sinusoidal(input_dim=3, output_dim=7, freq=np.zeros([5, 9]))
