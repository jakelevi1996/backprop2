""" This module contains error functions and their gradients, used with neural
networks to measure performance, and for training. The classes in this module
are private, and are exposed through public instances.

TODO:
-   Make BinaryCrossEntropy work with multiple data points and test
"""

import numpy as np
import plotting

class _ErrorFunction():
    """
    Template class for error functions which can be used to train arbitrary
    models
    """
    name = None

    def E(self, y, target):
        """
        Error function between y (predictions) and targets.

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
        First partial derivative of the error function, with respect to y
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
        Second partial derivative of the error function, with respect to y
        (predictions).

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
        Wrapper for the error function
        """
        return self.E(y, target)

    def get_id_from_func(self):
        """
        Given any child of the _ErrorFunction class, this method returns a
        unique integer ID for that child's class. This integer is used when
        saving models, to represent the activation function for a given layer in
        the model, and later used to restore the activation function using the
        get_func_from_id method
        """
        # Check this method is being called by a child of _ErrorFunction
        err_str = "Method must be called by a child of _ErrorFunction"
        if type(self) is _ErrorFunction:
            raise RuntimeError(err_str)
        # Get the list of subclasses of the _ErrorFunction class
        subclass_type_list = _ErrorFunction.__subclasses__()
        # Get the class of the object which is calling this method
        this_type = type(self)
        # Make sure that self is a child of _ErrorFunction
        if this_type not in subclass_type_list:
            raise RuntimeError("Invalid activation function")
        # Return a unique integer ID for the activation function
        return subclass_type_list.index(this_type)

    def plot(self, dir_name=".", xlims=[-5, 5], npoints=200):
        """
        Plot an activation function and its derivative, and save to disk
        """
        plotting.plot_error_func(self, dir_name, xlims, npoints)
    
def get_func_from_id(func_id):
    """
    Given an integer ID (EG generated using get_id_from_func), return an
    instance of the activation function which has the same unique integer ID.
    This method is used when loading models (after the activation functions are
    saved using their integer IDs), to restore the correct activation function
    for each layer in the network.

    TODO: should this be a class method? How do I use a class method?
    """
    # Get the list of subclasses of the _ErrorFunction class
    subclass_type_list = _ErrorFunction.__subclasses__()
    # Get the class of the activation function corresponding to id
    func_class = subclass_type_list[func_id]
    # Get an instance of the class
    func_object = func_class()
    # Return the instance of the correct activation function
    return func_object

class _SumOfSquares(_ErrorFunction):
    """ Sum of squares error function, used for regression, typically using a
    model with a linear activation function in the output layer """
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


class _BinaryCrossEntropy(_ErrorFunction):
    # TODO: this is not written for multi-dimensional inputs and outputs, or for
    # multiple data poitns
    def E(self, y, target):
        return - (target * np.log(y) + (1.0 - target) * np.log(1.0 - y)).sum()

    def dEdy(self, y, target):
        return ((y - target) / (y * (1.0 - y))).sum()

def _softmax(x):
    """ Apply a softmax function to the input x. It is assumed that x is a 2D
    matrix, in which dimension 0 of the input matrix refers to different
    dimensions of the input, and dimension 1 of the input matrix refers to
    different data points (so for example, a matrix containing 5 3D inputs would
    be represented by a 3x5 matrix), hence normalisation of the output is
    applied down dimension 0. """
    numerator = np.exp(x)
    return numerator / numerator.sum(axis=0, keepdims=True)

class _SoftmaxCrossEntropy(_ErrorFunction):
    """ This error function applies softmax and a cross entropy error function,
    typically used for training a model for multi-class classification. Softmax
    is applied here as part of the error function because it cannot be supported
    by the activations module, because it is not a 1D element-wise function (the
    normalisation of the output introduces dependency of each element in the
    output vector on all elements of the input vector), and hence is not
    compatible with the implementation of back-propagation in that module (and
    changing the implementation would likely reduce its efficiency).

    Because softmax is applied here, a multi-class classification model which is
    trained with this error function should have a linear activation function in
    the output layer. To apply a softmax function for prediction, the
    NeuralNetwork softmax_output_probabilities method should be used.

    Targets are assumed to be one-hot vectors (or matrices in the case of
    multiple data points) """

    name = "Softmax cross-entropy error function"

    def E(self, y, target):
        return -(target * np.log(_softmax(y))).sum(axis=0, keepdims=True)
    
    def dEdy(self, y, target):
        return _softmax(y) - target
    
    def d2Edy2(self, y, target):
        s = _softmax(y)
        hessian = (- np.expand_dims(s, 0)) * np.expand_dims(s, 1)
        diag_inds = np.arange(s.shape[0])
        hessian[diag_inds, diag_inds, :] += s
        return hessian

    def plot(self, dir_name=".", xlims=[-5, 5], npoints=200):
        t = np.zeros([2, npoints])
        y = np.zeros([2, npoints])
        t[1] = 1
        y[0] = np.linspace(*xlims, npoints)
        plotting.plot_error_func(self, dir_name, xlims, npoints, y, t)


# Expose public instances of private classes
sum_of_squares          = _SumOfSquares()
binary_cross_entropy    = _BinaryCrossEntropy()
softmax_cross_entropy   = _SoftmaxCrossEntropy()
