""" This module contains regulariser classes for the Dinosaur class for
meta-learning """

import numpy as np

class _Regulariser:
    """ Abstract parent class for regularisers """

    def __init__(self):
        self.mean = None
        self.parameter_scale = None
        self.error_scale = None

    def update(self, w_list, dE_list):
        """ Given a list of adapted weight-vectors for each task, and a list of
        reductions in the mean reconstruction error for each task, update the
        meta-parameters so that the meta-learning model fits the observed tasks
        more effectively, and update the regularisation error scaling
        coefficient so that the regularisation errors are scaled appropriately
        to the reconstruction errors """
        raise NotImplementedError()

    def get_error(self, w):
        """ Get the regularisation error for the given task-specific parameters
        and meta-parameters (mean, parameter scale and regularisation error
        scaling coefficient) """
        raise NotImplementedError()

    def get_gradient(self, w):
        """ Get the gradient of the regularisation function with respect to the
        task-specific parameters, for the given task-specific parameters and
        meta-parameters (mean, parameter scale and regularisation error scaling
        coefficient) """
        raise NotImplementedError()

    def _update_error_scale(self, dE_list):
        """ Update the regularisation error scaling coefficient so that the
        regularisation errors are scaled appropriately to the reconstruction
        errors. Specifically, the regularisation error is scaled such that
        during the next iteration of the meta-learning outer-loop, an increase
        of one in the unscaled regularisation error [after scaling] is equal to
        the mean reduction in reconstruction error during the current iteration
        of the meta-learning outer-loop.

        The regularisation error scaling coefficient is then rectified, such
        that if, during the current meta-learning outer-loop iteration, there
        has been an increase in the mean reduction in reconstruction error
        across tasks, then during the next iteration of the meta-learning outer
        loop, no regularisation error is applied. """
        self.error_scale = max(np.mean(dE_list), 0)


class Quadratic(_Regulariser):
    """ Regulariser with a quadratic error function. This regulariser is not
    expected to perform well, but is expected to be simple to implement, and
    serve as a benchmark for better regularisers """

    def update(self, w_list, dE_list):
        w_array = np.array(w_list)
        self.mean = np.mean(w_array, axis=0)
        self.parameter_scale = 1.0 / np.var(w_array - self.mean, axis=0)
        self._update_error_scale(dE_list)

    def get_error(self, w):
        error_unscaled = np.sum(
            np.square(w - self.mean) * self.parameter_scale
        )
        error = error_unscaled * self.error_scale
        return error

    def get_gradient(self, w):
        dEdw_unscaled = 2.0 * (w - self.mean) * self.parameter_scale
        dEdw = dEdw_unscaled * self.error_scale
        return dEdw

class Quartic(_Regulariser):
    """ Class for a multi-modal quartic regularisation function, which
    encourages parameters to move away from the mean in either direction
    towards a local optimum """

    def update(self, w_list, dE_list):
        w_array = np.array(w_list)
        self.mean = np.mean(w_array, axis=0)
        x2 = np.square(w_array - self.mean)
        x4 = np.square(x2)
        self.parameter_scale = np.sum(x2, axis=0) / np.sum(x4, axis=0)
        self._update_error_scale(dE_list)

    def get_error(self, w):
        x2 = np.square(w - self.mean)
        error_unscaled = np.square((self.parameter_scale * x2) - 1.0).sum()
        error = error_unscaled * self.error_scale
        return error

    def get_gradient(self, w):
        x = w - self.mean
        ax = self.parameter_scale * x
        ax2 = ax * x
        dEdw_unscaled = 4.0 * ax * (ax2 - 1.0)
        dEdw = dEdw_unscaled * self.error_scale
        return dEdw

regulariser_names_dict = {
    regulariser.__name__: regulariser
    for regulariser in _Regulariser.__subclasses__()
}
