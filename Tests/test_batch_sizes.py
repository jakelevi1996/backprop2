"""
Unit tests for classes in the optimisers.batch module, including
FullTrainingSet, ConstantBatchSize, and DynamicBatchSize
"""
import os
import numpy as np
import pytest
import optimisers, data, models
from .util import dataset_dict, get_random_network, get_output_dir
from .util import set_random_seed_from_args

# Get name of output directory, and create it if it doesn't exist
output_dir = get_output_dir("Batch sizes")

@pytest.mark.parametrize("dataset_key", dataset_dict.keys())
def test_full_training_set_batch(dataset_key):
    """
    Test using the full training set as a batch with the
    optimisers.batch.FullTrainingSet class
    """
    set_random_seed_from_args("test_full_training_set_batch", dataset_key)
    dataset = dataset_dict[dataset_key]
    batch_getter = optimisers.batch.FullTrainingSet()
    x_batch, y_batch = batch_getter.get_batch(dataset)
    assert x_batch.shape == dataset.x_train.shape
    assert y_batch.shape == dataset.y_train.shape

@pytest.mark.parametrize("dataset_key", dataset_dict.keys())
def test_constant_batch_size(dataset_key):
    """
    Test using a constant batch size with the optimisers.batch.ConstantBatchSize
    class
    """
    set_random_seed_from_args("test_constant_batch_size", dataset_key)
    dataset = dataset_dict[dataset_key]
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

@pytest.mark.parametrize("dataset_key", dataset_dict.keys())
def test_dynamic_batch_size(dataset_key):
    """
    Test using a dynamic batch size with the
    optimisers.batch.DynamicBatchSize class
    """
    set_random_seed_from_args("test_dynamic_batch_size", dataset_key)
    dataset = dataset_dict[dataset_key]
    batch_size = np.random.randint(10, 20)
    model = models.NeuralNetwork(dataset.input_dim, dataset.output_dim)
    batch_getter = optimisers.batch.DynamicBatchSize(model, dataset)
    n_iters = np.random.randint(50, 100)
    output_fname = "Test dynamic batch size, dataset = %s.txt" % dataset_key
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
