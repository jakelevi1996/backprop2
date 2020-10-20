"""
Module to contain activation functions, used in hidden and output layers of a
neural network. The classes in this module are private, and are exposed through
public instances.
"""
import numpy as np
import plotting


class _ActivationFunction():
    name = None

    def y(self, x):
        raise NotImplementedError

    def dydx(self, x):
        raise NotImplementedError

    def d2ydx2(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.y(x)

    def __repr__(self):
        return self.name

    def get_id_from_func(self):
        """
        Given any child of the _ActivationFunction class, this method returns a
        unique integer ID for that child's class. This integer is used when
        saving models, to represent the activation function for a given layer in
        the model, and later used to restore the activation function using the
        get_func_from_id method
        """
        # Check this method is being called by a child of _ActivationFunction
        err_str = "Method must be called by a child of _ActivationFunction"
        if type(self) is _ActivationFunction:
            raise RuntimeError(err_str)
        # Get the list of subclasses of the _ActivationFunction class
        subclass_type_list = _ActivationFunction.__subclasses__()
        # Get the class of the object which is calling this method
        this_type = type(self)
        # Make sure that self is a child of _ActivationFunction
        if this_type not in subclass_type_list:
            raise RuntimeError("Invalid activation function")
        # Return a unique integer ID for the activation function
        return subclass_type_list.index(this_type)
    
    def plot(self, dir_name=".", xlims=[-5, 5], npoints=200):
        """
        plot: plot an activation function and its derivative, and save to disk
        """
        plotting.plot_act_func(self, dir_name, xlims, npoints)

def get_func_from_id(func_id):
    """
    Given an integer ID (EG generated using
    _ActivationFunction.get_id_from_func), return an instance of the activation
    function which has the same unique integer ID. This function is used when
    loading models (after the activation functions are saved using their integer
    IDs), to restore the correct activation function for each layer in the
    network.
    """
    # Get the list of subclasses of the _ActivationFunction class
    subclass_type_list = _ActivationFunction.__subclasses__()
    # Get the class of the activation function corresponding to id
    func_class = subclass_type_list[func_id]
    # Get an instance of the class
    func_object = func_class()
    # Return the instance of the correct activation function
    return func_object

class _Identity(_ActivationFunction):
    name = "Identity activation function"

    def y(self, x):
        return x

    def dydx(self, x):
        return np.ones(x.shape)

    def d2ydx2(self, x):
        return np.zeros(x.shape)

class _Logistic(_ActivationFunction):
    name = "Logistic activation function"

    def y(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def dydx(self, x):
        y = self.y(x)
        return y * (1.0 - y)

    def d2ydx2(self, x):
        y = self.y(x)
        return y * (1 - y) * (1 - 2 * y)

class _Relu(_ActivationFunction):
    name = "ReLU activation function"

    def y(self, x):
        return np.where(x < 0.0, 0.0, x)

    def dydx(self, x):
        return np.where(x < 0.0, 0.0, 1.0)

    def d2ydx2(self, x):
        return np.where(x == 0.0, np.inf, 0.0)

        # return np.zeros(x.shape)

class _Gaussian(_ActivationFunction):
    name = "Gaussian activation function"

    def y(self, x):
        return np.exp(-(x ** 2))

    def dydx(self, x):
        y = self.y(x)
        return -2.0 * x * y

    def d2ydx2(self, x):
        y = self.y(x)
        return 2.0 * y * ((2.0 * x * x) - 1)

class _Cauchy(_ActivationFunction):
    name = "Cauchy activation function"

    def y(self, x):
        return 1.0 / (1.0 + x*x)

    def dydx(self, x):
        y = self.y(x)
        return -2.0 * x * y * y

    def d2ydx2(self, x):
        y = self.y(x)
        return 2.0 * y * y * ((4.0 * x * x * y) - 1)

class _SoftMax(_ActivationFunction):
    pass

class _PiecewiseQuadratic(_ActivationFunction):
    name = "Piecewise quadratic activation function"

    def y(self, x):
        xm2 = x - 2.0
        xp2 = x + 2.0
        y = np.where(x > 2.0,           0.0,                x)
        y = np.where(x < 2.0,           xm2 * xm2,          y)
        y = np.where(x < 1.0,           2.0 - x*x,          y)
        y = np.where(x < -1.0,          xp2 * xp2,          y)
        y = np.where(x < -2.0,          0.0,                y)
        return y

    def dydx(self, x):
        dydx = np.where(x > 2.0,        0.0,                x)
        dydx = np.where(x < 2.0,        2.0 * x - 4.0,      dydx)
        dydx = np.where(x < 1.0,        -2.0 * x,           dydx)
        dydx = np.where(x < -1.0,       2.0 * x + 4.0,      dydx)
        dydx = np.where(x < -2.0,       0.0,                dydx)
        return dydx

    def d2ydx2(self, x):
        d2ydx2 = np.where(x > 2.0,      0.0,                x)
        d2ydx2 = np.where(x < 2.0,      2.0,                d2ydx2)
        d2ydx2 = np.where(x < 1.0,      -2.0,               d2ydx2)
        d2ydx2 = np.where(x < -1.0,     2.0,                d2ydx2)
        d2ydx2 = np.where(x < -2.0,     0.0,                d2ydx2)
        return d2ydx2


# TODO: replace get_id_from_func and get_func_from_id functions with this dict
# or similar for saving/loading NeuralNetwork objects
act_func_names_dict = {
    act_func.__name__: act_func
    for act_func in _ActivationFunction.__subclasses__()
}

# Expose public instances of private classes
identity            = _Identity()
logistic            = _Logistic()
relu                = _Relu()
gaussian            = _Gaussian()
cauchy              = _Cauchy()
softMax             = _SoftMax()
piecewise_quadratic = _PiecewiseQuadratic()
