import os
import numpy as np

class DataSubset:
    def __init__(self, x=None, y=None, n=None):
        self.x = x
        self.y = y
        self.n = n

class DataSet:
    """ Interface class for data sets, which contains shape constants and
    train/test inputs and outputs as attributes, and methods for loading,
    saving, and printing the data.

    TODO: implement get_train_batch and get_test_batch and incorporate into
    optimisers module """
    def __init__(self):
        self.input_dim  = None
        self.output_dim = None
        self.train      = DataSubset()
        self.test       = DataSubset()

    def set_shape_constants(self, input_dim, output_dim, n_train, n_test):
        """ Set the input_dim, output_dim, n_train, and n_test attributes for
        this DataSet object, which determine the dimensionalities of the inputs
        and outputs, and the number of points in the training and test sets.

        Inputs:
        -   input_dim: dimensionality of inputs for this data set. Should be a
            positive integer
        -   output_dim: dimensionality of outputs for this data set. Should be
            a positive integer
        -   n_train: number of points in the training set for this data set.
            Should be None or a positive integer. If n_train is None, then it
            is chosen as 50 to the power of the input dimension if this is less
            than 5000, otherwise it is chosen as 5000
        -   n_test: number of points in the test set for this data set. Should
            be None or a positive integer. If n_test is None, then it is chosen
            to be equal to the number of points in the training set
        """
        if n_train is None:
            n_train = min(pow(50, input_dim), 5000)
        if n_test is None:
            n_test = n_train

        # Set shape constants
        self.input_dim      = input_dim
        self.output_dim     = output_dim
        self.train.n        = n_train
        self.test.n         = n_test


    def save(self, filename, dir_name="."):
        path = os.path.join(dir_name, filename + ".npz")
        np.savez(
            path,
            input_dim=self.input_dim,   output_dim=self.output_dim,
            n_train=self.train.n,       n_test=self.test.n,
            x_train=self.train.x,       x_test=self.test.x,
            y_train=self.train.y,       y_test=self.test.y,
        )

    def load(self, filename, dir_name="."):
        path = os.path.join(dir_name, filename + ".npz")
        # Load data from file
        with np.load(path) as data:
            self.input_dim              = data['input_dim']
            self.output_dim             = data['output_dim']
            self.train.n, self.test.n   = data['n_train'],  data['n_test']
            self.train.x, self.test.x   = data['x_train'],  data['x_test']
            self.train.y, self.test.y   = data['y_train'],  data['y_test']
        # Assert that the arrays have the correct shape
        assert self.train.x.shape == (self.input_dim    ,   self.train.n)
        assert self.train.y.shape == (self.output_dim   ,   self.train.n)
        assert self.test.x.shape  == (self.input_dim    ,   self.test.n )
        assert self.test.y.shape  == (self.output_dim   ,   self.test.n )

        return self

    def print_data(self, first_n=10, file=None):
        print(
            "train.x.T:",   self.train.x.T[:first_n],
            "train.y.T:",   self.train.y.T[:first_n],
            "test.x.T:",    self.test.x.T[:first_n],
            "test.y.T:",    self.test.y.T[:first_n],
            sep="\n",
            file=file
        )

    def __repr__(self):
        """ Return a string representation of this Dataset object, including
        the type of the object whose method is called, and the input and output
        dimensions, and number of points in the training and test sets """
        s = "%s(input_dim=%i, output_dim=%i, train.n=%i, test.n=%i)" % (
            type(self).__name__,
            self.input_dim,
            self.output_dim,
            self.train.n,
            self.test.n,
        )
        return s
