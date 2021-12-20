""" This module contains regulariser classes for the Dinosaur class for
meta-learning """

import numpy as np
import util
from optimisers.abstract_optimiser import AbstractOptimiser

class _Regulariser:
    """ Abstract parent class for regularisers """

    def __init__(self, error_scale_coefficient=1e-2):
        self.mean = None
        self.parameter_scale = None
        self.error_scale = None
        self._error_scale_coefficient = error_scale_coefficient

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
        self.error_scale = (
            self._error_scale_coefficient
        )


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

class QuarticType2(Quartic):
    """ Class for a quartic regulariser which sets the local optimum for
    task-specific parameters to the mean absolute distance of the parameter
    vector from the mean """

    def update(self, w_list, dE_list):
        w_array = np.array(w_list)
        self.mean = np.mean(w_array, axis=0)
        self.parameter_scale = np.clip(
            1.0 / np.mean(np.abs(w_array - self.mean), axis=0),
            None,
            1e4,
        )
        self._update_error_scale(dE_list)

class QuarticType3(Quartic):
    """ Class for a quartic regulariser which sets the local optimum for
    task-specific parameters to the RMS distance of the parameter vector from
    the mean """

    def update(self, w_list, dE_list):
        w_array = np.array(w_list)
        self.mean = np.mean(w_array, axis=0)
        self.parameter_scale = 1.0 / np.sqrt(np.mean(np.square(
            w_array - self.mean,
        ), axis=0))
        self._update_error_scale(dE_list)

class Eve(_Regulariser, AbstractOptimiser):
    """ Class for the Eve regulariser, which modifies the learning rate of each
    parameter according to the variance across tasks of the adapted
    task-specific values of that parameter (in such a way that parameters that
    vary significantly between tasks are given higher learning rates). This is
    the opposite of the Adam optimisation algorithm, which gives a higher
    learning rate to parameters that don't change very often. This approach is
    different to the other regularisation classes in this module, because there
    is no regularisation error function """

    def set_line_search(self, line_search):
        AbstractOptimiser.__init__(self, line_search)

    def update(self, w_list, _):
        w_array = np.array(w_list)
        self.mean = np.mean(w_array, axis=0)
        self.variance = np.var(w_array, axis=0)
        self.variance = np.clip(
            self.variance,
            a_min=(np.max(self.variance) / 1e6),
            a_max=None,
        )
        self.precision = 1.0 / self.variance
        self.learning_rate_scale = self.variance / np.mean(self.variance)
        self.parameter_scale = self.learning_rate_scale

    def _get_step(self, model, x_batch, y_batch):
        dEdw = model.get_gradient_vector(x_batch, y_batch)
        # Use stored _weight_vector attribute here? Needs to be made public
        w = model.get_parameter_vector()
        learning_rate = (
            self.learning_rate_scale
            * np.exp(-0.2 * self.precision * np.square(w - self.mean))
        )
        delta = -learning_rate * dEdw

        return delta, dEdw

    def get_convergence_metric(self, model):
        w = model.get_parameter_vector()
        distance_norm = np.sqrt(self.precision * np.square(w - self.mean))
        convergence_metric = np.mean(distance_norm)
        return convergence_metric

# Get dictionary mapping names of regularisers to the corresponding type
regulariser_names_dict = {
    regulariser.__name__: regulariser
    for regulariser in util.all_subclasses(_Regulariser)
}
