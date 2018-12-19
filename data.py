import numpy as np
import matplotlib.pyplot as plt

class DataSet():
    def __init__(self, filename=None):
        if filename is not None: self.load(filename)
        else:
            self.x_train = None
            self.y_train = None
            self.x_test = None
            self.y_test = None
    
    def save(self, filename):
        np.savez(
            filename, self.x_train, self.y_train, self.x_test, self.y_test
        )

    def load(self, filename):
        with np.load(filename) as loaded_data:
            self.x_train, self.y_train, self.x_test, self.y_test = [
                loaded_data[name] for name in loaded_data.files
            ]
    
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
    
    def print_data(self):
        print("x_train:\n", self.x_train)
        print("y_train:\n", self.y_train)
        print("x_test:\n", self.x_test)
        print("y_test:\n", self.y_test)

def sin(x, freq, ampl, phase, offset, noise_std):
    y = ampl * np.sin(2 * np.pi * freq * (x - phase)) + offset
    y += np.random.normal(scale=noise_std, size=x.shape)
    return  y

class SinusoidalDataSet(DataSet):
    def __init__(
        self, filename=None,
        n_train=100, n_test=50,
        freq=1.1, ampl=1.0, phase=0.1, offset=1.0,
        xlim=[0, 2.5], noise_std=0.1
    ):
        self.x_train = np.random.uniform(*xlim, size=n_train)
        self.x_test = np.random.uniform(*xlim, size=n_test)
        self.y_train = sin(self.x_train, freq, ampl, phase, offset, noise_std)
        self.y_test = sin(self.x_test, freq, ampl, phase, offset, noise_std)


def generate_sinusoidal_data():
    pass

def CircleDataSet(DataSet):
    pass

if __name__ == "__main__":
    s = SinusoidalDataSet(n_train=100, n_test=50, xlim=[0, 1])
    # s.print_data()
    filename = "Data/sinusoidal_data_set.npz"
    s.save(filename)

    sl = DataSet(filename)
    # sl.print_data()
    sl.plot("Data/sin")
