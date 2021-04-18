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
    
    def set_shape_constants(self, input_dim, output_dim, n_train, n_test):
        """ Set the input_dim, output_dim, n_train, and n_test attributes for
        this DataSet object, which determine the dimensionalities of the inputs
        and outputs, and the number of points in the training and test sets.

        Inputs:
        -   input_dim: dimensionality of inputs for this data set. Should be a
            positive integer
        -   output_dim: dimensionality of outputs for this data set. Should be a
            positive integer
        -   n_train: number of points in the training set for this data set.
            Should be None or a positive integer. If n_train is None, then it is
            chosen as 50 to the power of the input dimension
        -   n_test: number of points in the test set for this data set. Should
            be None or a positive integer. If n_test is None, then it is chosen
            to be equal to the number of points in the training set
        """
        if n_train is None:
            n_train = pow(50, input_dim)
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

class Regression(DataSet):
    """ Class for regression datasets. Outputs are continuous matrices with
    self.output_dim numbers of rows, and each column refers to a different data
    point. This class is used as a parent class for specific regression
    datasets. """

class Classification(DataSet):
    """ Class for classification datasets. Outputs are one-hot integer matrices,
    with self.output_dim number of rows (this is equal to the number of
    classes), and each column refers to a different data point. Each column has
    one value equal to 1, referring to which class that data point belongs to,
    and the rest of the values are equal to zero. This class is used as a parent
    class for specific regression datasets. """

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


class Sinusoidal(Regression):
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
        # Set shape constants and unspecified parameters
        self.set_shape_constants(input_dim, output_dim, n_train, n_test)
        if phase is None:
            phase = np.random.normal(size=[input_dim, 1])
        if freq is None:
            freq = np.random.normal(size=[output_dim, input_dim])
        if ampl is None:
            ampl = np.random.normal(size=[output_dim, output_dim])
        if offset is None:
            offset = np.random.normal(size=[output_dim, 1])
        # Generate input/output training and test data
        self.x_train = np.random.uniform(
            x_lo,
            x_hi,
            size=[input_dim, self.n_train]
        )
        self.x_test  = np.random.uniform(
            x_lo,
            x_hi,
            size=[input_dim, self.n_test]
        )
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

class MixtureOfGaussians(Classification):
    """ Class for a mixture-of-Gaussians classification dataset. The means of
    the mixture components are generated from a normal distribution, and the
    variance matrices are generated randomly and implicitly. There can be more
    mixture components than classes, although by default there will be there
    same number of mixture components and classes.  """
    def __init__(
        self,
        input_dim=2,
        output_dim=3,
        n_train=None,
        n_test=None,
        n_mixture_components=None,
        scale=0.2,
    ):
        """ Initialise a mixture-of-Gaussians classification data set.

        Inputs:
        -   input_dim: dimensionality of inputs for this data set. Should be a
            positive integer
        -   output_dim: dimensionality of outputs for this data set. Should be a
            positive integer
        -   n_train: number of points in the training set for this data set.
            Should be None or a positive integer. If n_train is None, then it is
            chosen as 50 to the power of the input dimension
        -   n_test: number of points in the test set for this data set. Should
            be None or a positive integer. If n_test is None, then it is chosen
            to be equal to the number of points in the training set
        -   n_mixture_components: number of mixture components in the data set.
            Should be None, or a positive integer. Can be more than the output
            dimension, in which case some classes will have multiple mixture
            components, or less than the output dimension, in which case some
            classes will have no mixture components (and therefore no data
            points)
        -   scale: positive float, which determines the expected variance of the
            mixture components relative to the distance between them. A larger
            scale will mean that the variances of mixture components are large,
            and therefore the mixture components are more likely to overlap and
            become more difficult to distinguish, making the classification task
            "harder"
        """
        # Set shape constants and number of mixture components
        self.set_shape_constants(input_dim, output_dim, n_train, n_test)
        if n_mixture_components is None:
            n_mixture_components = self.output_dim
        
        # Generate mean and scale for each mixture component
        mean = np.random.normal(size=[n_mixture_components, input_dim])
        scale_matrix = scale * np.random.normal(
            size=[n_mixture_components, input_dim, input_dim]
        )
        # Generate all of the input points
        self.x_train = np.random.normal(size=[self.input_dim, self.n_train])
        self.x_test  = np.random.normal(size=[self.input_dim, self.n_test])
        # Generate assignments of inputs to mixture components
        z_train = np.random.randint(n_mixture_components, size=self.n_train)
        z_test  = np.random.randint(n_mixture_components, size=self.n_test)
        # Transform each input point according to its mixture component
        for i in range(n_mixture_components):
            self.x_train[:, z_train == i] = (
                (scale_matrix[i] @ self.x_train[:, z_train == i])
                + mean[i].reshape(-1, 1)
            )
            self.x_test[:, z_test == i] = (
                (scale_matrix[i] @ self.x_test[:, z_test == i])
                + mean[i].reshape(-1, 1)
            )
        # Initialise mixture_to_class which maps mixture components to classes
        if n_mixture_components > output_dim:
            mixture_to_class = np.full(n_mixture_components, np.nan)
            initial_class_assignment_inds = np.random.choice(
                n_mixture_components,
                size=output_dim,
                replace=False,
            )
            mixture_to_class[initial_class_assignment_inds] = np.arange(
                output_dim
            )
            mixture_to_class[np.isnan(mixture_to_class)] = np.random.randint(
                output_dim,
                size=n_mixture_components-output_dim,
            )
            mixture_to_class = mixture_to_class.astype(int)
        elif n_mixture_components < output_dim:
            mixture_to_class = np.random.choice(
                output_dim,
                size=n_mixture_components,
                replace=False,
            )
        else:
            mixture_to_class = np.arange(output_dim)
        # Set labels and one-hot output data
        self.train_labels = mixture_to_class[z_train]
        self.test_labels  = mixture_to_class[z_test]
        self.y_train = np.zeros([output_dim, self.n_train])
        self.y_test  = np.zeros([output_dim, self.n_test])
        self.y_train[self.train_labels, np.arange(self.n_train)] = 1
        self.y_test[ self.test_labels,  np.arange(self.n_test)]  = 1


class CircleDataSet(DataSet):
    pass

class SumOfGaussianCurvesDataSet(DataSet):
    pass

class GaussianCurveDataSet(DataSet):
    """ Wrapper for SumOfGaussianCurvesDataSet """
    pass
