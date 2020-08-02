import numpy as np
import plotting


class ActivationFunction():
    name = None

    def y(self, x):         raise NotImplementedError
    def dydx(self, x):      raise NotImplementedError
    def d2ydx2(self, x):    raise NotImplementedError
    def __call__(self, x):  return self.y(x)

    def get_id_from_func(self):
        """
        get_id_from_func: given any child of the ActivationFunction class, this
        method returns a unique integer ID for that child's class. This integer
        is used when saving models, to represent the activation function for a
        given layer in the model, and later used to restore the activation
        function using the get_func_from_id method
        """
        # Check this method is being called by a child of ActivationFunction
        err_str = "Method must be called by a child of ActivationFunction"
        if type(self) is ActivationFunction: raise RuntimeError(err_str)
        # Get the list of subclasses of the ActivationFunction class
        subclass_type_list = ActivationFunction.__subclasses__()
        # Get the class of the object which is calling this method
        this_type = type(self)
        # Make sure that self is a child of ActivationFunction
        if this_type not in subclass_type_list:
            raise RuntimeError("Invalid activation function")
        # Return a unique integer ID for the activation function
        return subclass_type_list.index(this_type)
    
    def get_func_from_id(self, func_id):
        """
        get_func_from_id: given an integer ID (EG generated using
        get_id_from_func), return an instance of the activation function which
        has the same unique integer ID. This method is used when loading models
        (after the activation functions are saved using their integer IDs), to
        restore the correct activation function for each layer in the network 
        """
        # Get the list of subclasses of the ActivationFunction class
        subclass_type_list = ActivationFunction.__subclasses__()
        # Get the class of the activation function corresponding to id
        func_class = subclass_type_list[func_id]
        # Get an instance of the class
        func_object = func_class()
        # Return the instance of the correct activation function
        return func_object

    def plot(self, dir_name=".", xlims=[-5, 5], npoints=200):
        """
        plot: plot an activation function and its derivative, and save to disk
        """
        plotting.plot_act_func(self, dir_name, xlims, npoints)

class Identity(ActivationFunction):
    name = "Identity activation function"
    def y(self, x): return x
    def dydx(self, x): return np.ones(x.shape)

class Logistic(ActivationFunction):
    name = "Logistic activation function"
    def y(self, x): return 1.0 / (1.0 + np.exp(-x))
    def dydx(self, x, y=None):
        if y is None: y = self.y(x)
        return y * (1.0 - y)

class Relu(ActivationFunction):
    name = "ReLU activation function"
    def y(self, x): return np.where(x < 0.0, 0.0, x)
    def dydx(self, x): return np.where(x < 0.0, 0.0, 1.0)

class Gaussian(ActivationFunction):
    name = "Gaussian activation function"
    def y(self, x): return np.exp(-(x ** 2))
    def dydx(self, x, y=None):
        if y is None: y = self.y(x)
        return -2.0 * x * y

class SoftMax(ActivationFunction):
    pass

if __name__ == "__main__":
    a = Logistic()
    id = a.get_id_from_func()
    b = ActivationFunction().get_func_from_id(id)
    print(a, b, a(3), b(3), sep="\n")    

    Identity().plot("Data/Identity activation function")
    Logistic().plot("Data/Logistic activation function")
    Gaussian().plot("Data/Gaussian activation function")
    Relu().plot("Data/ReLU activation function")
