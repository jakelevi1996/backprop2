""" This module contains regulariser classes for the Dinosaur class """

class _Regulariser:
    """ Abstract parent class for regularisers """

    def update(self, w_list):
        """ Given a list of adapted weight-vectors, update the meta-parameters
        so that the meta-learning model fits the observed tasks more
        effectively """
        raise NotImplementedError()

    def get_error(self, w, mean, scale):
        """ Get the regularisation error for the given task-specific parameters
        and meta-parameters (mean and scale) """
        raise NotImplementedError()

    def get_gradient(self, w, mean, scale):
        """ Get the gradient of the regularisation function with respect to the
        task-specific parameters, for the given task-specific parameters and
        meta-parameters (mean and scale) """
        raise NotImplementedError()

class Quadratic(_Regulariser):
    """ Regulariser with a quadratic error function. This regulariser is not
    expected to perform well, but is expected to be simple to implement, and
    serve as a benchmark for better regularisers """
