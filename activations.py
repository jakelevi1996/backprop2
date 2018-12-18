import numpy as np
import matplotlib.pyplot as plt

class ActivationFunction():
    name = None
    def __call__(self, x): raise NotImplementedError
    def dydx(self, x): raise NotImplementedError
    def plot(self, filename, xlims=[-5, 5]):
        x = np.linspace(*xlims, 200)
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
    def __call__(self, x): return 1.0 / (1.0 + np.exp(-x))
    def dydx(self, x):
        logit = self.__call__(x)
        return logit * (1.0 - logit)

class Identity(ActivationFunction):
    name = "Identity activation function"
    def __call__(self, x): return x
    def dydx(self, x): return np.ones(x.shape)

class Gaussian(ActivationFunction):
    name = "Gaussian activation function"
    def __call__(self, x): return np.exp(-(x ** 2))
    def dydx(self, x): return - 2.0 * x * np.exp(-(x ** 2))


if __name__ == "__main__":
    a = Logistic()
    print(a(3))
    Logistic().plot("Data/Logistic activation function")
    Identity().plot("Data/Identity activation function")
    Gaussian().plot("Data/Gaussian activation function")