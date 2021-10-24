from data.regression import Regression
import numpy as np

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
            numpy array with shape [input_dim, 1]. Default is randomly
            generated
        -   freq: frequency used in sinusoidal function. Should be a float, or
            a numpy array with shape that broadcasts to [output_dim,
            input_dim]. Default is randomly generated
        -   ampl: amplitude used in sinusoidal function. Should be a float, or
            a numpy array with shape that broadcasts to [output_dim,
            output_dim]. Default is randomly generated
        -   offset: offset used in sinusoidal function. Should be a float, or a
            numpy array with shape [output_dim, 1]. Default is randomly
            generated

        Outputs:
        -   Sinusoidal DataSet object initialised with noisy training and test
            data

        Raises:
        -   ValueError: if x-limits x_lo and x_hi don't broadcast to the size
            of train.x and test.x
        """
        Regression.__init__(self)
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
        self.train.x = np.random.uniform(
            x_lo,
            x_hi,
            size=[input_dim, self.train.n],
        )
        self.test.x  = np.random.uniform(
            x_lo,
            x_hi,
            size=[input_dim, self.test.n],
        )
        self.train.y = noisy_sin(
            self.train.x,
            phase,
            freq,
            ampl,
            offset,
            noise_std,
            output_dim,
        )
        self.test.y = noisy_sin(
            self.test.x,
            phase,
            freq,
            ampl,
            offset,
            noise_std,
            output_dim,
        )
