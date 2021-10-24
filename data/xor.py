from data.classification import BinaryClassification
import numpy as np


class Xor(BinaryClassification):
    """ Class for a binary logical XOR dataset. Input coordinates are generated
    randomly and uniformly between -1 and 1 """
    def __init__(
        self,
        input_dim=2,
        n_train=None,
        n_test=None,
    ):
        """ Initialise a binary XOR classification data set.

        Inputs:
        -   input_dim: dimensionality of inputs for this data set. Should be a
            positive integer
        -   n_train: number of points in the training set for this data set.
            Should be None or a positive integer. If n_train is None, then it
            is chosen as 50 to the power of the input dimension
        -   n_test: number of points in the test set for this data set. Should
            be None or a positive integer. If n_test is None, then it is chosen
            to be equal to the number of points in the training set
        """
        BinaryClassification.__init__(self)
        # Set shape constants
        self.set_shape_constants(input_dim, 1, n_train, n_test)

        # Generate input data
        self.train.x = np.random.uniform(
            low=-1,
            high=1,
            size=[self.input_dim, self.train.n],
        )
        self.test.x = np.random.uniform(
            low=-1,
            high=1,
            size=[self.input_dim, self.test.n],
        )

        # Generate output labels
        self.train.y    = (self.train.x > 0).sum(axis=0, keepdims=True) % 2
        self.test.y     = (self.test.x  > 0).sum(axis=0, keepdims=True) % 2
