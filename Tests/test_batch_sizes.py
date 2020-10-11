"""
Unit tests for classes in the optimisers.batch module, including
FullTrainingSet, ConstantBatchSize, and DynamicBatchSize
"""
import numpy as np
import pytest
import optimisers, data

dataset_list = [
    data.SinusoidalDataSet1D1D(),
    data.SinusoidalDataSet2DnD(nx0=17, nx1=11, output_dim=2),
    data.SinusoidalDataSet2DnD(nx0=10, nx1=3, output_dim=4),
]

@pytest.mark.parametrize("seed", [5802, 5496, 5922])
@pytest.mark.parametrize("dataset", dataset_list)
def test_full_training_set_batch(seed, dataset):
    """
    Test using the full training set as a batch with the
    optimisers.batch.FullTrainingSet class
    """
    np.random.seed(seed)
    batch_getter = optimisers.batch.FullTrainingSet()
    x_batch, y_batch = batch_getter.get_batch(dataset)
    assert x_batch.shape == dataset.x_train.shape
    assert y_batch.shape == dataset.y_train.shape

@pytest.mark.parametrize("seed", [5802, 5496, 5922])
@pytest.mark.parametrize("dataset", dataset_list)
def test_constant_batch_size(seed, dataset):
    """
    Test using a constant batch size with the optimisers.batch.ConstantBatchSize
    class
    """
    np.random.seed(seed)
    batch_size = np.random.randint(10, 20)
    batch_getter = optimisers.batch.ConstantBatchSize(batch_size)
    x_batch, y_batch = batch_getter.get_batch(dataset)
    assert x_batch.shape == (dataset.x_train.shape[0], batch_size)
    assert y_batch.shape == (dataset.y_train.shape[0], batch_size)

@pytest.mark.parametrize("seed", [5802, 5496, 5922])
@pytest.mark.parametrize("dataset", dataset_list)
def test_dynamic_batch_size(seed, dataset):
    """
    Test using a dynamic batch size with the
    optimisers.batch.DynamicBatchSize class
    """
    pass
