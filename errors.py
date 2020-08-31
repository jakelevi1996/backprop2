"""
Module to for error functions, used with neural network to measure performance,
and for training.

TODO:
-   Make BinaryCrossEntropy work with multiple data points
-   Second derivatives
-   Softmax error function
"""

import numpy as np
import plotting

class ErrorFunction():
    """
    ErrorFunction: template class for error functions which can be used to train
    arbitrary models
    """
    name = None

    def E(self, y, target):
        """
        E: error function between y (predictions) and targets.

        Inputs:
        -   y: predictions from a model. Should be in a numpy array with shape
            (output_dim, N_D)
        -   target: targets that the model is trying to match. Should be in a
            numpy array with shape (output_dim, N_D)

        Outputs:
        -   error: array containing the error for each data point, in a numpy
            array with shape (1, N_D)
        """
        raise NotImplementedError

    def dEdy(self, y, target):
        """
        dEdy: first partial derivative of the error function, with respect to y
        (predictions).

        Inputs:
        -   y: predictions from a model. Should be in a numpy array with shape
            (output_dim, N_D)
        -   target: targets that the model is trying to match. Should be in a
            numpy array with shape (output_dim, N_D)

        Outputs:
        -   error_grad: array containing the first partial derivative of the
            error with respect to the predictions for each data point, in a
            numpy array with shape (output_dim, N_D)
        """
        raise NotImplementedError

    def d2Edy2(self, y, target):
        """
        d2Edy2: second partial derivative of the error function, with respect to
        y (predictions).

        Inputs:
        -   y: predictions from a model. Should be in a numpy array with shape
            (output_dim, N_D)
        -   target: targets that the model is trying to match. Should be in a
            numpy array with shape (output_dim, N_D)

        Outputs:
        -   error_hessian: array containing the second partial derivative of the
            error with respect to the predictions for each data point, in a
            numpy array with shape (output_dim, output_dim, N_D)

        TODO: if the hessian is symetrical, can save time and memory by only
        calculating half? Or is it diagonal, in which case store in a
        (output_dim, N_D) array?
        """
        raise NotImplementedError

    def __call__(self, y, target):
        """
        __call__: wrapper for the error function
        """
        return self.E(y, target)

    def get_id_from_func(self):
        """
        get_id_from_func: given any child of the ErrorFunction class, this
        method returns a unique integer ID for that child's class. This integer
        is used when saving models, to represent the activation function for a
        given layer in the model, and later used to restore the activation
        function using the get_func_from_id method
        """
        # Check this method is being called by a child of ErrorFunction
        err_str = "Method must be called by a child of ErrorFunction"
        if type(self) is ErrorFunction: raise RuntimeError(err_str)
        # Get the list of subclasses of the ErrorFunction class
        subclass_type_list = ErrorFunction.__subclasses__()
        # Get the class of the object which is calling this method
        this_type = type(self)
        # Make sure that self is a child of ErrorFunction
        if this_type not in subclass_type_list:
            raise RuntimeError("Invalid activation function")
        # Return a unique integer ID for the activation function
        return subclass_type_list.index(this_type)

    def plot(self, dir_name=".", xlims=[-5, 5], npoints=200):
        """
        plot: plot an activation function and its derivative, and save to disk
        """
        plotting.plot_error_func(self, dir_name, xlims, npoints)
    
def get_func_from_id(func_id):
    """
    get_func_from_id: given an integer ID (EG generated using
    get_id_from_func), return an instance of the activation function which
    has the same unique integer ID. This method is used when loading models
    (after the activation functions are saved using their integer IDs), to
    restore the correct activation function for each layer in the network.

    TODO: should this be a class method? How do I use a class method?
    """
    # Get the list of subclasses of the ErrorFunction class
    subclass_type_list = ErrorFunction.__subclasses__()
    # Get the class of the activation function corresponding to id
    func_class = subclass_type_list[func_id]
    # Get an instance of the class
    func_object = func_class()
    # Return the instance of the correct activation function
    return func_object

class SumOfSquares(ErrorFunction):
    """
    SumOfSquares: sum of squares error function, used for regression
    """
    name = "Sum of squares error function"
    def E(self, y, target):
        return 0.5 * np.square(y - target).sum(axis=0, keepdims=True)
    
    def dEdy(self, y, target):
        return y - target
    
    def d2Edy2(self, y, target):
        # Get the output dimension and number of data points
        output_dim, N_D = y.shape
        # Create identity matrix with the correct rank
        result = np.identity(output_dim)
        # Expand dimensions to have a 3rd axis
        result = np.expand_dims(result, axis=2)
        # Repeat for each data point and return
        return np.repeat(result, N_D, axis=2)
        # TODO: will this work if returning a scalar 1, due to broadcasting?


class BinaryCrossEntropy(ErrorFunction):
    # TODO: this is not written for multi-dimensional inputs and outputs, or for
    # multiple data poitns
    def E(self, y, target):
        return - (target * np.log(y) + (1.0 - target) * np.log(1.0 - y)).sum()

    def dEdy(self, y, target):
        return ((y - target) / (y * (1.0 - y))).sum()

class Softmax(ErrorFunction):
    # TODO
    pass
