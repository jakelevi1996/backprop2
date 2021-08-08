from data.classification import BinaryClassification
import numpy as np

class Disk(BinaryClassification):
    """ Class for a binary classification dataset consisting of a disk. Input
    coordinates are generated randomly and uniformly between -1 and 1 """
    def __init__(
        self,
        input_dim=2,
        n_train=None,
        n_test=None,
    ):
        """ Initialise a binary disk classification data set.

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
        # Set shape constants
        self.set_shape_constants(input_dim, 1, n_train, n_test)

        # Generate input data
        self.x_train = np.random.uniform(
            low=-1,
            high=1,
            size=[self.input_dim, self.n_train],
        )
        self.x_test = np.random.uniform(
            low=-1,
            high=1,
            size=[self.input_dim, self.n_test],
        )

        # Generate output labels
        x_train_radius  = np.square(self.x_train).sum(axis=0, keepdims=True)
        x_test_radius   = np.square(self.x_test).sum(axis=0, keepdims=True)
        self.y_train    = np.logical_and(
            x_train_radius > 0.5,
            x_train_radius < 0.75,
            dtype=int,
        )
        self.y_test     = np.logical_and(
            x_test_radius > 0.5,
            x_test_radius < 0.75,
            dtype=int,
        )
