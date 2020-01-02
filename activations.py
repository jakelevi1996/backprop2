import numpy as np
import plotting


class ActivationFunction():
    id      = None
    name    = None

    def get_act_func(self, id):
        """
        get_act_func: return the activation function corresponding to the input
        id. This function is used for loading neural networks, when the
        activation functions are saved using their integer IDs
        """
        for act_func in [Identity, Logistic, Relu, Gaussian, SoftMax]:
            if id == act_func.id: return act_func()
        raise ValueError("Invalid activation function ID")

    def y(self, x):         raise NotImplementedError
    def dydx(self, x):      raise NotImplementedError
    def d2ydx2(self, x):    raise NotImplementedError
    def __call__(self, x):  return self.y(x)

    def plot(self, filename, xlims=[-5, 5], npoints=200):
        """
        plot: plot an activation function and its derivative, and save
        """
        plotting.plot_act_func(filename, self, xlims, npoints)

class Identity(ActivationFunction):
    id      = 0
    name    = "Identity activation function"
    def y(self, x): return x
    def dydx(self, x): return np.ones(x.shape)

class Logistic(ActivationFunction):
    id      = 1
    name    = "Logistic activation function"
    def y(self, x): return 1.0 / (1.0 + np.exp(-x))
    def dydx(self, x, y=None):
        if y is None: y = self.y(x)
        return y * (1.0 - y)

class Relu(ActivationFunction):
    id      = 2
    name    = "ReLU activation function"
    def y(self, x):
        f = np.zeros(x.shape)
        f[x > 0] = x[x > 0]
        return f
    def dydx(self, x):
        f = np.zeros(x.shape)
        f[x > 0] = 1.0
        return f

class Gaussian(ActivationFunction):
    id      = 3
    name    = "Gaussian activation function"
    def y(self, x): return np.exp(-(x ** 2))
    def dydx(self, x, y=None):
        if y is None: y = self.y(x)
        return - 2.0 * x * y

class SoftMax(ActivationFunction):
    id      = 4
    pass

if __name__ == "__main__":
    a = Logistic()
    print(a(3))
    Identity().plot("Data/Identity activation function")
    Logistic().plot("Data/Logistic activation function")
    Gaussian().plot("Data/Gaussian activation function")
    Relu().plot("Data/ReLU activation function")
    act_func = ActivationFunction().get_act_func(1)
    act_func.plot("Data/Act func id = 1")