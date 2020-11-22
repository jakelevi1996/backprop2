"""
Module for creating DataSet objects, with input and output data, training and
test partitions, and methods for batching, saving, loading, and printing the
data.

TODO: implement DataSet classes for CircleDataSet, SumOfGaussianCurvesDataSet,
GaussianCurveDataSet. One module per class, moved over to the data directory?

TODO: have a class/subclasses for generating input points (EG uniform, Gaussian,
grid, etc), which is shared between all data classes?
"""
import os
import numpy as np

class DataSet():
    """
    Interface class for data sets, which contains shape constants and train/test
    inputs and outputs as attributes, and methods for loading, saving, and
    printing the data.

    TODO: implement get_train_batch and get_test_batch and incorporate into
    optimisers module
    """
    def __init__(self, filename=None, dir_name="."):
        # If a filename is specified, then load from file
        if filename is not None:
            self.load(filename, dir_name)
        else:
            self.input_dim  , self.output_dim   = None, None
            self.n_train    , self.n_test       = None, None
            self.x_train    , self.y_train      = None, None
            self.x_test     , self.y_test       = None, None
    
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

def noisy_sin(x, phase, freq, ampl, offset, noise_std, output_dim):
    """
    noisy_sin: apply a linearly transformed sinusoidal function to a linearly
    transformed set of input data, and add Gaussian noise.

    Inputs:
    -   x: input data. Should be a np.ndarray with shape [input_dim, N_D]
    -   phase: constant which is added to each dimension of the input data.
        Should be either a scalar or a np.ndarray with shape [input_dim, 1]
    -   freq: linear rescaling of input data dimensions. Should be either a
        scalar or a np.ndarray with shape [output_dim, input_dim]
    -   ampl: linear transformation which is applied to the output of the
        sinusoidal function. Should be either a scalar or a np.ndarray with
        shape [output_dim, output_dim]
    -   offset: constant which is added to the linearly transformed output from
        the sinusoidal function. Should be either a scalar or a np.ndarray with
        shape [output_dim, 1]
    -   noise_std: standard deviation of the noise which is added to the final
        output. Should be either a scalar or a np.ndarray with shape
        [output_dim, 1]

    Outputs:
    -   y: output data, in a np.ndarray with shape [output_dim, N_D]
    """
    y = np.dot(ampl, np.sin(2 * np.pi * np.dot(freq, (x - phase))))
    return y + np.random.normal(offset, noise_std, [output_dim, x.shape[1]])


class Sinusoidal(DataSet):
    """
    Class for a sinusoidal data set. The input and output dimensions are
    configurable through the initialiser arguments. The training and test set
    inputs are uniformaly distributed between specified limits.
    """
    def __init__(
        self,
        input_dim=1,
        output_dim=1,
        n_train=None,
        n_test=None,
        x_lo=-2,
        x_hi=2,
        noise_std=0.1,
        phase=None,
        freq=None,
        ampl=None,
        offset=None,
    ):
        """
        Initialise a noisy Sinusoidal dataset object.

        Inputs:
        -   input_dim: integer number of input dimensions. Default is 1
        -   output_dim: integer number of output dimensions. Default is  1
        -   n_train: integer number of points in the training set. Default is
            100 ^ input_dim
        -   n_test: integer number of points in the test set. Default is the
            same as n_train
        -   x_lo: the lower limit for random uniformly generated x-values.
            Should be a float, or a numpy array with shape [input_dim, 1].
            Default is -2
        -   x_hi: the upper limit for random uniformly generated x-values.
            Should be a float, or a numpy array with shape [input_dim, 1].
            Default is 2
        -   noise_std: standard deviation for Gaussian noise applied to output
            data. Should be a float, or a numpy array with shape [output_dim,
            1]. Default is 0.1
        -   phase: phase used in sinusoidal function. Should be a float, or a
            numpy array with shape [input_dim, 1]. Default is randomly generated
        -   freq: frequency used in sinusoidal function. Should be a float, or a
            numpy array with shape that broadcasts to [output_dim, input_dim].
            Default is randomly generated
        -   ampl: amplitude used in sinusoidal function. Should be a float, or a
            numpy array with shape that broadcasts to [output_dim, output_dim].
            Default is randomly generated
        -   offset: offset used in sinusoidal function. Should be a float, or a
            numpy array with shape [output_dim, 1]. Default is randomly
            generated

        Outputs:
        -   Sinusoidal DataSet object initialised with noisy training and test
            data

        Raises:
        -   ValueError: if x-limits x_lo and x_hi don't broadcast to the size of
            x_train and x_test
        """
        # Set unspecified parameters
        if n_train is None:
            n_train = pow(100, input_dim)
        if n_test is None:
            n_test = n_train
        if phase is None:
            phase = np.random.normal(size=[input_dim, 1])
        if freq is None:
            freq = np.random.normal(size=[output_dim, input_dim])
        if ampl is None:
            ampl = np.random.normal(size=[output_dim, output_dim])
        if offset is None:
            offset = np.random.normal(size=[output_dim, 1])
        # Set shape constants
        self.input_dim  , self.output_dim   = input_dim , output_dim
        self.n_train    , self.n_test       = n_train   , n_test
        # Generate input/output training and test data
        self.x_train = np.random.uniform(x_lo, x_hi, size=[input_dim, n_train])
        self.x_test  = np.random.uniform(x_lo, x_hi, size=[input_dim, n_test])
        self.y_train = noisy_sin(
            self.x_train,
            phase,
            freq,
            ampl,
            offset,
            noise_std,
            output_dim
        )
        self.y_test = noisy_sin(
            self.x_test,
            phase,
            freq,
            ampl,
            offset,
            noise_std,
            output_dim
        )

class CircleDataSet(DataSet):
    pass

class SumOfGaussianCurvesDataSet(DataSet):
    pass

class GaussianCurveDataSet(DataSet):
    """ Wrapper for SumOfGaussianCurvesDataSet """
    pass
