import os
import numpy as np
import pytest
import data as d

# Define list of activation functions to be tested
dataset_list = [d.SinusoidalDataSet1D1D, d.SinusoidalDataSet2DnD]

# Get name of output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs")

@pytest.mark.parametrize("seed", [4405, 9721, 5974])
@pytest.mark.parametrize("dataset_subclass", dataset_list)
def test_save_load(seed, dataset_subclass):
    """
    Test initialising a dataset subclass with default constructor arguments,
    check that it can be saved and loaded, and that the saved and loaded dataset
    objects are equivalent
    """
    np.random.seed(seed)
    dataset = dataset_subclass()
    dataset.save(dataset_subclass.__name__, output_dir)
    dataset_loaded = d.DataSet(dataset_subclass.__name__, output_dir)
    attr_list = [
        "input_dim", "output_dim", "n_train", "n_test",
        "x_train", "y_train", "x_test", "y_test"
    ]
    for a in attr_list:
        assert np.all(getattr(dataset, a) == getattr(dataset_loaded, a))

@pytest.mark.parametrize("seed", [1854, 7484, 5736])
@pytest.mark.parametrize("dataset_subclass", dataset_list)
def test_print_data(seed, dataset_subclass):
    np.random.seed(seed)
    dataset = dataset_subclass()
    # Print data to stdout
    dataset.print_data()
    # Print data to file
    filename = dataset_subclass.__name__ + "data.txt"
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        dataset.print_data(file=f)
    


# # Generate 1D to 1D sinusoidal regression dataset
# s11 = d.SinusoidalDataSet1D1D(n_train=100, n_test=50, xlim=[0, 1])
# filename = "Data/sin_dataset_11.npz"
# s11.save(filename)
# s_load = d.DataSet(filename)
# s_load.print_data()

# # Test 2D to 3D sinusoidal regression dataset
# s23 = d.SinusoidalDataSet2DnD(nx0=23, nx1=56, output_dim=3)
# filename = "Data/sin_dataset_23.npz"
# s23.save(filename)
# sl = d.DataSet(filename)
# sl.print_data()
