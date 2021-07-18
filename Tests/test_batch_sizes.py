""" Unit tests for classes in the optimisers.batch module, including
FullTrainingSet, ConstantBatchSize, and DynamicBatchSize """
import os
import numpy as np
import pytest
import optimisers, data, models
from .util import (
    get_output_dir,
    set_random_seed_from_args,
    get_random_dataset,
    get_random_network,
)

# Get name of output directory, and create it if it doesn't exist
output_dir = get_output_dir("Batch sizes")

@pytest.mark.parametrize("repeat", range(3))
def test_full_training_set_batch(repeat):
    """ Test using the full training set as a batch with the
    optimisers.batch.FullTrainingSet class """
    set_random_seed_from_args("test_full_training_set_batch", repeat)
    dataset = get_random_dataset()
    batch_getter = optimisers.batch.FullTrainingSet()
    x_batch, y_batch = batch_getter.get_batch(dataset)
    assert x_batch.shape == dataset.x_train.shape
    assert y_batch.shape == dataset.y_train.shape

@pytest.mark.parametrize("repeat", range(3))
def test_constant_batch_size(repeat):
    """ Test using a constant batch size with the
    optimisers.batch.ConstantBatchSize class """
    set_random_seed_from_args("test_constant_batch_size", repeat)
    dataset = get_random_dataset()
    batch_size = np.random.randint(10, 20)
    batch_getter = optimisers.batch.ConstantBatchSize(batch_size)
    x_batch, y_batch = batch_getter.get_batch(dataset)
    assert x_batch.shape == (dataset.x_train.shape[0], batch_size)
    assert y_batch.shape == (dataset.y_train.shape[0], batch_size)

def test_constant_batch_size_non_integer_fail():
    """ Test that using a non-integer batch size raises an exception when using
    the optimisers.batch.ConstantBatchSize class """
    batch_size = 3.7
    with pytest.raises(TypeError):
        batch_getter = optimisers.batch.ConstantBatchSize(batch_size)
