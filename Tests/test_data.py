import os
import numpy as np
import pytest
import data
from .util import (
    get_output_dir,
    set_random_seed_from_args,
    get_dataset_and_name_from_type,
)

# Get name of output directory, and create it if it doesn't exist
output_dir = get_output_dir("Data")

@pytest.mark.parametrize("repeat", range(3))
@pytest.mark.parametrize("dataset_type", data.dataset_class_dict.values())
def test_save_load(dataset_type, repeat):
    """ Test initialising a dataset subclass with a random number of input and
    output dimensions and number of points in the training and test sets, and
    check that it can be saved and loaded, and that the saved and loaded
    dataset objects are equivalent """
    set_random_seed_from_args("test_save_load", dataset_type, repeat)
    dataset, dataset_name = get_dataset_and_name_from_type(dataset_type)
    dataset.save(dataset_name, output_dir)
    dataset_loaded = data.DataSet(dataset_name, output_dir)
    attr_list = [
        "input_dim" , "output_dim"  ,
        "n_train"   , "n_test"      ,
        "x_train"   , "x_test"      ,
        "y_train"   , "y_test"      ,
    ]
    for a in attr_list:
        assert np.all(getattr(dataset, a) == getattr(dataset_loaded, a))

@pytest.mark.parametrize("repeat", range(3))
@pytest.mark.parametrize("dataset_type", data.dataset_class_dict.values())
def test_print_data(dataset_type, repeat):
    """ Test the print_data method of an instance of a Dataset subclass """
    set_random_seed_from_args("test_print_data", dataset_type, repeat)
    dataset, dataset_name = get_dataset_and_name_from_type(dataset_type)
    # Print data to stdout
    dataset.print_data()
    # Print data to file
    filename = dataset_name + " data.txt"
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        dataset.print_data(file=f)

def test_sinusoidal_valid_xlim():
    """ Test initialising a Sinusoidal DataSet with valid single- and
    multi-dimensional x-limits """
    # Test single-dimensional x-limits
    d = data.Sinusoidal(input_dim=4, x_hi=7.5, n_train=10)

    # Test multi-dimensional x-limits
    x_hi = np.array([3, 4, 5, 6]).reshape(-1, 1)
    d = data.Sinusoidal(input_dim=4, x_hi=x_hi, n_train=10)

def test_sinusoidal_invalid_xlim():
    """ Test that a ValueError is raised when initialising a Sinusoidal DataSet
    with x-limits that don't broadcast to the size of x_train and x_test.

    As stated in the docstring for the data.Sinusoidal initialiser, x_hi
    "should be a float, or a numpy array with shape [input_dim, 1]". """
    with pytest.raises(ValueError):
        d = data.Sinusoidal(input_dim=3, x_lo=[1, 2])
    
    with pytest.raises(ValueError):
        d = data.Sinusoidal(input_dim=4, x_hi=[3, 4, 5, 6], n_train=10)
    
    with pytest.raises(ValueError):
        x_hi = np.array([3, 4, 5, 6]).reshape(1, -1)
        d = data.Sinusoidal(input_dim=4, x_hi=x_hi, n_train=10)

def test_sinusoidal_invalid_freq_shape():
    """ Test that initialising a Sinusoidal DataSet with a frequency that
    doesn't broadcast to the input and output dimensions raises a ValueError
    """
    with pytest.raises(ValueError):
        d = data.Sinusoidal(input_dim=3, output_dim=7, freq=np.zeros([5, 9]))

@pytest.mark.parametrize("n_mixture_components", [2, 4, 6])
def test_gaussian_mixture_num_components(n_mixture_components):
    """ Test initialising a mixture-of-Gaussians classification dataset where
    the number of mixture components is less than, equal to, and greater than
    the output dimension (IE the number of classes) """
    # Set random seed
    set_random_seed_from_args(
        "test_gaussian_mixture_num_components",
        n_mixture_components,
    )
    # Initialise input arguments
    output_dim  = 4
    n_train     = np.random.randint(10, 20)
    n_test      = np.random.randint(10, 20)
    input_dim   = np.random.randint(2, 5)
    # Initialise data set
    classification_data = data.MixtureOfGaussians(
        input_dim=input_dim,
        output_dim=output_dim,
        n_train=n_train,
        n_test=n_test,
        n_mixture_components=n_mixture_components,
    )
    assert classification_data.x_train.shape        == (input_dim, n_train)
    assert classification_data.x_test.shape         == (input_dim, n_test)
    assert classification_data.train_labels.shape   == (n_train, )
    assert classification_data.test_labels.shape    == (n_test, )
    assert classification_data.y_train.shape        == (output_dim, n_train)
    assert classification_data.y_test.shape         == (output_dim, n_test)

@pytest.mark.parametrize("n_mixture_components", [2, 5])
def test_binary_gaussian_mixture_num_components(n_mixture_components):
    """ Test initialising a binary mixture-of-Gaussians classification dataset
    where the number of mixture components is equal to and greater
    than the number of classes (IE 2) """
    # Set random seed
    set_random_seed_from_args(
        "test_binary_gaussian_mixture_num_components",
        n_mixture_components,
    )
    # Initialise input arguments
    n_train     = np.random.randint(10, 20)
    n_test      = np.random.randint(10, 20)
    input_dim   = np.random.randint(2, 5)
    # Initialise data set
    classification_data = data.BinaryMixtureOfGaussians(
        input_dim=input_dim,
        n_train=n_train,
        n_test=n_test,
        n_mixture_components=n_mixture_components,
    )
    assert classification_data.x_train.shape        == (input_dim, n_train)
    assert classification_data.x_test.shape         == (input_dim, n_test)
    assert classification_data.y_train.shape        == (1, n_train)
    assert classification_data.y_test.shape         == (1, n_test)
