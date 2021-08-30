""" This module contains regulariser classes for the Dinosaur class for
meta-learning """

import numpy as np

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

    def update(self, w_list):
        self.mean = np.mean(w_list, axis=0)
        self.scale = np.var(np.array(w_list) - self.mean, axis=0)
    
    def get_error(self, w, mean, scale):
        error = (np.square(w - self.mean) / self.scale).sum()
        return error
    
    def get_gradient(self, w, mean, scale):
        dEdw = 2.0 * (w - self.mean) / self.scale
        return dEdw
