"""
Module to contain batch-getters, which are classes containing a get_batch
method, called in AbstractOptimiser.optimise; this method accepts a dataset
object, and returns x_batch and y_batch, which are taken from the dataset object
and used as the inputs and outputs for one iteration of training a model.

TODO:
-   Add docstrings
-   When using random choice to select batch_inds, compare with replacement vs
    without replacement
-   Rename DynamicBatchSize as LocalDynamicBatchSize, and implement a
    GlobalDynamicBatchSize, which calculates the probability that the dot
    product of the descent direction with the gradient vector is < 0 (IE
    guaranteeing a reduction). The dot product is a linear combination of each
    element of the descent direction, and the random variables representing each
    element of the gradient vector; assuming independence, a mean and variance
    can be calculated for the entire dot product, and by calculating the
    probability of this dot product being <= 0, it should be possible to
    calculate a batch size. In this scenario, where does N (the number of data
    points/batch size) come into the picture? Same as before, averaging over N
    samples, variance scales by 1/N

"""
import numpy as np
from scipy.stats import norm
from optimisers.terminator import Terminator

class _BatchGetter():
    """ Abstract parent class for batch-getters, containing the get_batch
    method. This class should be subclassed by public subclasses, which
    implement different strategies for choosing a batch from a data-set"""
    def get_batch(self, dataset):
        """ Get a batch of data, used for one iteration of training. This method
        is called by AbstractOptimiser.optimise """
        raise NotImplementedError()

class FullTrainingSet(_BatchGetter):
    """ Class for a batch-getter which returns the full training set as a batch.
    This is useful when the size of the data-set is small, EG on the order of
    100 data-points, especially with data-sets that have one-dimensional inputs
    """
    def get_batch(self, dataset):
        return dataset.x_train, dataset.y_train

class ConstantBatchSize(_BatchGetter):
    """ Class for a batch-getter which returns a randomly-selected batch of a
    constant size. This size is specified when this object is initialised. """
    def __init__(self, batch_size, replace=True):
        if type(batch_size) is not int:
            raise TypeError("batch_size argument must be an integer")

        self.batch_size = batch_size
        self.replace = replace
    
    def get_batch(self, dataset):
        batch_inds = np.random.choice(
            dataset.n_train,
            size=self.batch_size,
            replace=self.replace
        )
        return dataset.x_train[:, batch_inds], dataset.y_train[:, batch_inds]

class DynamicBatchSize(_BatchGetter, Terminator):
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
