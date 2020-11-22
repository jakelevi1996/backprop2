"""
Unit tests for classes in the optimisers.batch module, including
FullTrainingSet, ConstantBatchSize, and DynamicBatchSize
"""
import os
import numpy as np
import pytest
import optimisers, data, models
from .util import get_dataset, dataset_list, get_random_network, get_output_dir

# Get name of output directory, and create it if it doesn't exist
output_dir = get_output_dir("Batch sizes")

@pytest.mark.parametrize("seed", [5802, 5496, 5922])
@pytest.mark.parametrize("dataset_str", dataset_list)
def test_full_training_set_batch(seed, dataset_str):
    """
    Test using the full training set as a batch with the
    optimisers.batch.FullTrainingSet class
    """
    np.random.seed(seed)
    dataset = get_dataset(dataset_str)
    batch_getter = optimisers.batch.FullTrainingSet(dataset)
    x_batch, y_batch = batch_getter.get_batch(dataset)
    assert x_batch.shape == dataset.x_train.shape
    assert y_batch.shape == dataset.y_train.shape

@pytest.mark.parametrize("seed", [5802, 5496, 5922])
@pytest.mark.parametrize("dataset_str", dataset_list)
def test_constant_batch_size(seed, dataset_str):
    """
    Test using a constant batch size with the optimisers.batch.ConstantBatchSize
    class
    """
    np.random.seed(seed)
    dataset = get_dataset(dataset_str)
    batch_size = np.random.randint(10, 20)
    batch_getter = optimisers.batch.ConstantBatchSize(batch_size)
    x_batch, y_batch = batch_getter.get_batch(dataset)
    assert x_batch.shape == (dataset.x_train.shape[0], batch_size)
    assert y_batch.shape == (dataset.y_train.shape[0], batch_size)

def test_constant_batch_size_non_integer_fail():
    """
    Test that using a non-integer batch size raises an exception when using the
    optimisers.batch.ConstantBatchSize class
    """
    batch_size = 3.7
    with pytest.raises(TypeError):
        batch_getter = optimisers.batch.ConstantBatchSize(batch_size)

def test_invalid_xlim():
    """
    Test that initialising a DataSet with x-limits that don't broadcast to the
    size of x_train and x_test raises a ValueError
    """
    with pytest.raises(ValueError):
        d = data.Sinusoidal(input_dim=3, x_lo=[1, 2])
    
    with pytest.raises(ValueError):
        d = data.Sinusoidal(input_dim=4, x_hi=[3, 4, 5, 6])

@pytest.mark.parametrize(
    "seed, dataset_str",
    zip([5802, 5496, 5922, 8948], dataset_list)
)
def test_dynamic_batch_size(seed, dataset_str):
    """
    Test using a dynamic batch size with the
    optimisers.batch.DynamicBatchSize class
    """
    np.random.seed(seed)
    dataset = get_dataset(dataset_str)
    batch_size = np.random.randint(10, 20)
    model = models.NeuralNetwork(dataset.input_dim, dataset.output_dim)
    batch_getter = optimisers.batch.DynamicBatchSize(model, dataset)
    n_iters = np.random.randint(50, 100)
    output_fname = "Test dynamic batch size, dataset = %s.txt" % dataset_str
    with open(os.path.join(output_dir, output_fname), "w") as f:
        result = optimisers.Result(file=f)
        result.add_column(optimisers.results.columns.BatchSize(batch_getter))
        result = optimisers.gradient_descent(
            model,
            dataset,
            result=result,
            batch_getter=batch_getter,
            terminator=optimisers.Terminator(i_lim=n_iters),
            evaluator=optimisers.Evaluator(i_interval=1)
        )
