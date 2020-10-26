"""
TODO

TODO:
- Add docstrings
- When using random choice to select batch_inds, compare with replacement vs
  without replacement
"""
import numpy as np
from scipy.stats import norm
from _optimisers.terminator import Terminator

class BatchGetter():
    def get_batch(self, dataset):
        raise NotImplementedError

class FullTrainingSet(BatchGetter):
    def __init__(self, dataset):
        self.batch_size = dataset.n_train

    def get_batch(self, dataset):
        return dataset.x_train, dataset.y_train

class ConstantBatchSize(BatchGetter):
    def __init__(self, batch_size):
        if type(batch_size) is not int:
            raise TypeError("batch_size argument must be an integer")

        self.batch_size = batch_size
    
    def get_batch(self, dataset):
        batch_inds = np.random.choice(dataset.n_train, size=self.batch_size)
        return dataset.x_train[:, batch_inds], dataset.y_train[:, batch_inds]

class DynamicBatchSize(BatchGetter, Terminator):
    def __init__(
        self,
        model,
        dataset,
        prob_correct_direction=0.99,
        alpha_smooth=0.2,
        init_batch_size=50,
        min_batch_size=10,
    ):
        """
        ...

        TODO: use isf instead of ppf, update kwarg name accordingly and update
        tests and scripts (or alternatively, allow to specify the scale
        directly?)
        """
        inv_cdf_p               = norm.ppf(prob_correct_direction)
        self.scale              = inv_cdf_p * inv_cdf_p
        self.model              = model
        self.batch_size         = init_batch_size
        self.min_batch_size     = min_batch_size
        self.max_batch_size     = dataset.n_train
        self.alpha_smooth       = alpha_smooth
        self.one_m_alpha_smooth = 1 - alpha_smooth
        # Call get_gradient_vector once so gradients will be initialised
        batch_inds = np.random.choice(dataset.n_train, size=init_batch_size)
        model.get_gradient_vector(
            dataset.x_train[:, batch_inds],
            dataset.y_train[:, batch_inds]
        )


    def get_batch(self, dataset):
        """
        Get dynamically calculated batch size.

        TODO:
        -   Have option to only recalculate batch size every N iterations
        -   Use Kalman filter instead of exponential smoothing
        """
        # Calculate new batch size and smooth with current batch size
        new_batch_size = self.model.get_dbs_metric() * self.scale
        self.batch_size *= self.alpha_smooth
        self.batch_size += self.one_m_alpha_smooth * new_batch_size
        # Clip batch-size to [self.min_batch_size, dataset.n_train]
        self.batch_size = min(
            max(self.batch_size, self.min_batch_size),
            dataset.n_train
        )
        # Get batch inds and extract batch data from dataset
        batch_inds = np.random.choice(
            dataset.n_train,
            size=int(self.batch_size)
        )
        return dataset.x_train[:, batch_inds], dataset.y_train[:, batch_inds]

    def ready_to_terminate(self, **kwargs):
        return self.batch_size >= self.max_batch_size
