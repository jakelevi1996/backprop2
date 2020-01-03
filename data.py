import numpy as np

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
    
    def get_train_batch(self, batch_size):  raise NotImplementedError
    
    def get_test_batch(self, batch_size):   raise NotImplementedError
    
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
            "x_train.T:",   self.x_train.T[:first_n],
            "y_train.T:",   self.y_train.T[:first_n],
            "x_test.T:",    self.x_test.T[:first_n],
            "y_test.T:",    self.y_test.T[:first_n], sep="\n"
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

class SinusoidalDataSet1D1D(DataSet):
    """
    SinusoidalDataSet1D1D: class for a sinusoidal data set with 1D inputs and 1D
    outputs, and sensible default values for phase, frequency, amplitude and
    offset
    """
    def __init__(
        self, n_train=100, n_test=50, phase=0.1, freq=1.1,
        ampl=1.0, offset=1.0, xlim=[-2, 2], noise_std=0.1
    ):
        # Set shape constants
        self.input_dim  , self.output_dim   = 1         , 1
        self.n_train    , self.n_test       = n_train   , n_test
        # Generate input/output training and test data
        self.x_train    = np.random.uniform(*xlim, size=[1, n_train])
        self.x_test     = np.random.uniform(*xlim, size=[1, n_test])
        self.y_train    = noisy_sin(
            self.x_train, phase, freq, ampl, offset, noise_std, 1
        )
        self.y_test     = noisy_sin(
            self.x_test, phase, freq, ampl, offset, noise_std, 1
        )

class SinusoidalDataSet2DnD(DataSet):
    """
    SinusoidalDataSet2DnD: class for a sinusoidal data set with 2D inputs and nD
    outputs. The test set is a uniform mesh of points (which makes plotting
    easier), and the training set is a random subset of these.

    TODO: the parameters of the random generation of phase, frequency, amplitude
    and offset may or may not need to be altered to give more sensible typical
    values
    """
    def __init__(
        self, nx0=200, x0lim=[-2, 2], nx1=200, x1lim=[-2, 2],
        noise_std=0.1, train_ratio=0.8, output_dim=3
    ):
        input_dim = 2
        # Generate test set inputs as a uniform mesh and reshape
        x0, x1 = np.linspace(*x0lim, nx0), np.linspace(*x1lim, nx1)
        x0_mesh, x1_mesh = np.meshgrid(x0, x1)
        self.x_test = np.stack([x0_mesh.ravel(), x1_mesh.ravel()], axis=0)
        n_test = self.x_test.shape[1]
        # Randomly generate phase, frequency, amplitude, and offset
        phase   = np.random.normal(size=[input_dim, 1])
        freq    = np.random.normal(size=[output_dim, input_dim])
        ampl    = np.random.normal(size=[output_dim, output_dim])
        offset  = np.random.normal(size=[output_dim, 1])
        # Generate noiseless test set outputs
        self.y_test = noisy_sin(
            self.x_test, phase, freq, ampl, offset, 0, output_dim
        )
        # Generate training set as a random subset of the test set
        n_train = int(n_test * train_ratio)
        train_inds = np.random.choice(n_test, n_train, replace=False)
        self.x_train = self.x_test[:, train_inds]
        self.y_train = self.y_test[:, train_inds]
        # Add noise to training and test outputs independently
        self.y_test += np.random.normal(0, noise_std, self.y_test.shape)
        self.y_train += np.random.normal(0, noise_std, self.y_train.shape)
        # Set shape constants
        self.input_dim  , self.output_dim   = input_dim , output_dim
        self.n_train    , self.n_test       = n_train   , n_test
        self.nx0        , self.nx1          = nx0       , nx1
        self.train_inds = train_inds


def generate_sinusoidal_data():
    pass

class CircleDataSet(DataSet):
    pass

class SumOfGaussianCurvesDataSet(DataSet):
    pass

class GaussianCurveDataSet(DataSet):
    # Wrapper for SumOfGaussianCurvesDataSet
    pass

if __name__ == "__main__":
    np.random.seed(0)

    # Generate 1D to 1D sinusoidal regression dataset
    s11 = SinusoidalDataSet1D1D(n_train=100, n_test=50, xlim=[0, 1])
    filename = "Data/sin_dataset_11.npz"
    s11.save(filename)
    s_load = DataSet(filename)
    s_load.print_data()

    # Test 2D to 3D sinusoidal regression dataset
    s23 = SinusoidalDataSet2DnD(nx0=23, nx1=56, output_dim=3)
    filename = "Data/sin_dataset_23.npz"
    s23.save(filename)
    sl = DataSet(filename)
    sl.print_data()
