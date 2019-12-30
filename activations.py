import numpy as np
import matplotlib.pyplot as plt

class ActivationFunction():
    name = None
    def y(self, x): raise NotImplementedError
    def dydx(self, x): raise NotImplementedError
    def d2ydx2(self, x): raise NotImplementedError
    def __call__(self, x): return self.y(x)
    def plot(self, filename, xlims=[-5, 5], npoints=200):
        x = np.linspace(*xlims, npoints)
        y = self.__call__(x)
        dydx = self.dydx(x)
        plt.figure(figsize=[8, 6])
        plt.plot(x, y, 'b', x, dydx, 'r', alpha=0.75)
        plt.legend(["Function", "Derivative"])
        plt.title(self.name)
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

class Logistic(ActivationFunction):
    name = "Logistic activation function"
    def y(self, x): return 1.0 / (1.0 + np.exp(-x))
    def dydx(self, x, y=None):
        if y is None: y = self.y(x)
        return y * (1.0 - y)

class Identity(ActivationFunction):
    name = "Identity activation function"
    def y(self, x): return x
    def dydx(self, x): return np.ones(x.shape)

class Relu(ActivationFunction):
    name = "ReLU activation function"
    def y(self, x):
        f = np.zeros(x.shape)
        f[x > 0] = x[x > 0]
        return f
    def dydx(self, x):
        f = np.zeros(x.shape)
        f[x > 0] = 1.0
        return f

class Gaussian(ActivationFunction):
    name = "Gaussian activation function"
    def y(self, x): return np.exp(-(x ** 2))
    def dydx(self, x, y=None):
        if y is None: y = self.y(x)
        return - 2.0 * x * y

class SoftMax(ActivationFunction):
    pass

if __name__ == "__main__":
    a = Logistic()
    print(a(3))
    Logistic().plot("Data/Logistic activation function")
    Identity().plot("Data/Identity activation function")
    Gaussian().plot("Data/Gaussian activation function")
    Relu().plot("Data/ReLU activation function")