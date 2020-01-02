import numpy as np

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

class SumOfSquares(ErrorFunction):
    """
    SumOfSquares: sum of squares error function, used for regression
    """
    name = "sum of squares"
    def E(self, y, target):
        return 0.5 * np.square(y - target).sum(axis=0, keepdims=True)
    
    def dEdy(self, y, target):
        return y - target
    
    def d2Edy2(self, y, target):
        # NB: for sum-of-squares, this is just the identity matrix
        raise NotImplementedError

class BinaryCrossEntropy(ErrorFunction):
    # TODO: this is not written for multi-dimensional inputs and outputs, or for
    # multiple data poitns
    def E(self, y, target):
        return - (target * np.log(y) + (1.0 - target) * np.log(1.0 - y)).sum()

    def dEdy(self, y, target):
        return ((y - target) / (y * (1.0 - y))).sum()
