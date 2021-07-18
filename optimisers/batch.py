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
            replace=self.replace,
        )
        return dataset.x_train[:, batch_inds], dataset.y_train[:, batch_inds]
