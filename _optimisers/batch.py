"""
TODO

TODO:
- Add docstrings
- When using random choice to select batch_inds, compare with replacement vs
  without replacement
"""
import numpy as np

class BatchGetter():
    def get_batch(self, dataset, *args):
        raise NotImplementedError

class FullTrainingSet(BatchGetter):
    # def get_batch(self, dataset, *_):
    def get_batch(self, dataset, *args):
        return dataset.x_train, dataset.y_train

class ConstantBatchSize(BatchGetter):
    def __init__(self, batch_size):
        self._batch_size = batch_size
    
    def get_batch(self, dataset, *args):
        batch_inds = np.random.choice(dataset.n_train, size=self._batch_size)
        return dataset.x_train[:, batch_inds], dataset.y_train[:, batch_inds]

class DynamicBatchSize(BatchGetter):
    def __init__(self):
        raise NotImplementedError

    def get_batch(self, dataset, *args):
        raise NotImplementedError
