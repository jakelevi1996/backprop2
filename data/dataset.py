import os
import numpy as np

class DataSet:
    """ Interface class for data sets, which contains shape constants and
    train/test inputs and outputs as attributes, and methods for loading,
    saving, and printing the data.

    TODO: implement get_train_batch and get_test_batch and incorporate into
    optimisers module """
    def __init__(self, filename=None, dir_name="."):
        # If a filename is specified, then load from file
        if filename is not None:
            self.load(filename, dir_name)
        else:
            self.input_dim  , self.output_dim   = None, None
            self.n_train    , self.n_test       = None, None
            self.x_train    , self.y_train      = None, None
            self.x_test     , self.y_test       = None, None
    
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
        self.n_train        = n_train
        self.n_test         = n_test

    
    def save(self, filename, dir_name="."):
        path = os.path.join(dir_name, filename + ".npz")
        np.savez(
            path,
            input_dim=self.input_dim,   output_dim=self.output_dim,
            n_train=self.n_train,       n_test=self.n_test,
            x_train=self.x_train,       x_test=self.x_test,
            y_train=self.y_train,       y_test=self.y_test
        )

    def load(self, filename, dir_name="."):
        path = os.path.join(dir_name, filename + ".npz")
        # Load data from file
        with np.load(path) as data:
            self.input_dim              = data['input_dim']
            self.output_dim             = data['output_dim']
            self.n_train, self.n_test   = data['n_train'],  data['n_test']
            self.x_train, self.x_test   = data['x_train'],  data['x_test']
            self.y_train, self.y_test   = data['y_train'],  data['y_test']
        # Assert that the arrays have the correct shape
        assert self.x_train.shape == (self.input_dim    ,   self.n_train)
        assert self.y_train.shape == (self.output_dim   ,   self.n_train)
        assert self.x_test.shape  == (self.input_dim    ,   self.n_test )
        assert self.y_test.shape  == (self.output_dim   ,   self.n_test )
    
    def print_data(self, first_n=10, file=None):
        print(
            "x_train.T:",   self.x_train.T[:first_n],
            "y_train.T:",   self.y_train.T[:first_n],
            "x_test.T:",    self.x_test.T[:first_n],
            "y_test.T:",    self.y_test.T[:first_n],
            sep="\n",
            file=file
        )
    
    def __repr__(self):
        """ Return a string representation of this Dataset object, including
        the type of the object whose method is called, and the input and output
        dimensions, and number of points in the training and test sets """
        s = "%s(input_dim=%i, output_dim=%i, n_train=%i, n_test=%i)" % (
            type(self).__name__,
            self.input_dim,
            self.output_dim,
            self.n_train,
            self.n_test,
        )
        return s
