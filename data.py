import numpy as np
import matplotlib.pyplot as plt

class DataSet():
    """
    Interface class for data sets, which contains shape constants and train/test
    inputs and outputs as attributes, and methods for loading, saving, and
    printing the data.
    """
    def __init__(self, filename=None):
        # If a filename is specified, then load from file
        if filename is not None: self.load(filename)
        else:
            self.input_dim  , self.output_dim   = None, None
            self.n_train    , self.n_test       = None, None
            self.x_train    , self.y_train      = None, None
            self.x_test     , self.y_test       = None, None
    
    def save(self, filename):
        np.savez(
            filename, input_dim=self.input_dim, output_dim=self.output_dim,
            n_train=self.n_train, n_test=self.n_test,
            x_train=self.x_train, x_test=self.x_test,
            y_train=self.y_train, y_test=self.y_test
        )

    def load(self, filename):
        # Load data from file
        with np.load(filename) as data:
            self.input_dim              = data['input_dim']
            self.output_dim             = data['output_dim']
            self.n_train, self.n_test   = data['n_train'], data['n_test']
            self.x_train, self.x_test   = data['x_train'], data['x_test']
            self.y_train, self.y_test   = data['y_train'], data['y_test']
        # Assert that the arrays have the correct shape
        assert self.x_train.shape == (self.input_dim , self.n_train)
        assert self.x_test.shape  == (self.input_dim , self.n_test )
        assert self.y_train.shape == (self.output_dim, self.n_train)
        assert self.y_test.shape  == (self.output_dim, self.n_test )
    
    def print_data(self, first_n=10):
        print(
            "x_train.T:", self.x_train.T[:first_n],
            "y_train.T:", self.y_train.T[:first_n],
            "x_test.T:", self.x_test.T[:first_n],
            "y_test.T:", self.y_test.T[:first_n], sep="\n"
        )

    # TODO: plotting functions in the plotting module
    def plot(self, filename, figsize=[8, 6], title="Regression data"):
        plt.figure(figsize=figsize)
        plt.plot(self.x_train, self.y_train, "bo", alpha=0.75)
        plt.plot(self.x_test, self.y_test, "ro", alpha=0.75)
        plt.title(title)
        plt.legend(["Training data", "Testing data"])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    def scatter(self, filename, figsize=[8, 6]):
        pass
        # TODO: add plotting method for binary/discrete data

def noisy_sin(x, phase, freq, ampl, offset, noise_std, output_dim):
    """
    noisy_sin: apply a linearly transformed sinusoidal function to a linearly
    transformed set of input data, and add Gaussian noise.

    Inputs:
    -   x: input data. Should be a np.ndarray with shape [input_dim, N_D]
    -   phase: constant which is added to each dimension of the input data.
        Should be either a scalar or a np.ndarray with shape [input_dim, 1]
    -   freq: linear rescaling of input data dimensions. Should be either a
        scalar or a np.ndarray with shape [input_dim, 1]
    -   ampl: linear transformation which is applied to the output of the
        sinusoidal function. Should be either a scalar or a np.ndarray with
        shape [output_dim, input_dim]
    -   offset: constant which is added to the linearly transformed output from
        the sinusoidal function. Should be either a scalar or a np.ndarray with
        shape [output_dim, 1]
    -   noise_std: standard deviation of the noise which is added to the final
        output. Should be either a scalar or a np.ndarray with shape
        [output_dim, 1]
    
    Outputs:
    -   y: output data, in a np.ndarray with shape [output_dim, N_D]
    """
    y = np.dot(ampl, np.sin(2 * np.pi * freq * (x - phase)))
    return y + np.random.normal(offset, noise_std, [output_dim, x.shape[1]])

class SinusoidalDataSet11(DataSet):
    """
    SinusoidalDataSet11: class for a sinusoidal data set with 1D inputs and 1D
    outputs, and sensible default values for phase, frequency, amplitude and
    offset
    """
    def __init__(
        self, filename=None, n_train=100, n_test=50, phase=0.1, freq=1.1,
        ampl=1.0, offset=1.0, xlim=[-2, 2], noise_std=0.1
    ):
        # Set shape constants
        self.input_dim  , self.output_dim   = 1         , 1
        self.n_train    , self.n_test       = n_train   , n_test
        # Generate input/output training and test data
        self.x_train = np.random.uniform(*xlim, size=[1, n_train])
        self.x_test = np.random.uniform(*xlim, size=[1, n_test])
        self.y_train = noisy_sin(
            self.x_train, phase, freq, ampl, offset, noise_std, 1
        )
        self.y_test = noisy_sin(
            self.x_test, phase, freq, ampl, offset, noise_std, 1
        )

class SinusoidalDataSet2n(DataSet):
    """
    SinusoidalDataSet2n: class for a sinusoidal data set with 2D inputs and nD
    outputs. The test set is a uniform mesh of points (which makes plotting
    easier), and the training set is a random subset of these.

    TODO: the parameters of the random generation of phase, frequency, amplitude
    and offset may or may not need to be altered to give more sensible typical
    values
    """
    def __init__(
        self, filename=None, nx0=200, x0lim=[-2, 2], nx1=200, x1lim=[-2, 2],
        noise_std=0.1, n_train=200, output_dim=3
    ):
        input_dim = 2
        # Generate test set inputs as a uniform mesh and reshape
        x0, x1 = np.linspace(*x0lim, nx0), np.linspace(*x1lim, nx1)
        x0_mesh, x1_mesh = np.meshgrid(x0, x1)
        self.x_test = np.stack([x0_mesh.ravel(), x1_mesh.ravel()], axis=0)
        n_test = self.x_test.shape[1]
        # Randomly generate phase, frequency, amplitude, and offset
        phase   = np.random.normal(size=[input_dim, 1])
        freq    = np.random.normal(size=[input_dim, 1])
        ampl    = np.random.normal(size=[output_dim, input_dim])
        offset  = np.random.normal(size=[output_dim, 1])
        # Generate test set outputs
        self.y_test = noisy_sin(
            self.x_test, phase, freq, ampl, offset, noise_std, output_dim
        )
        # Generate training set as a random subset of the test set
        train_inds = np.random.choice(n_test, n_train)
        self.x_train = self.x_test[:, train_inds]
        self.y_train = self.y_test[:, train_inds]
        # Set shape constants
        self.input_dim  , self.output_dim   = input_dim , output_dim
        self.n_train    , self.n_test       = n_train   , n_test
        self.nx0        , self.nx1          = nx0       , nx1


def generate_sinusoidal_data():
    pass

def CircleDataSet(DataSet):
    pass

if __name__ == "__main__":
    np.random.seed(0)
    s11 = SinusoidalDataSet11(n_train=100, n_test=50, xlim=[0, 1])
    # s11.print_data()
    filename = "Data/sin_dataset_11.npz"
    s11.save(filename)

    sl = DataSet(filename)
    # sl.print_data()
    sl.plot("Data/sin")

    s23 = SinusoidalDataSet2n(output_dim=3)
    filename = "Data/sin_dataset_23.npz"
    s23.save(filename)
    sl = DataSet(filename)
    sl.print_data()

