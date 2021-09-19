import numpy as np
from optimisers.timer import TimedObject
from optimisers.batch import _BatchGetter
from optimisers.smooth import MovingAverage

class Terminator(TimedObject):
    """ The Terminator class is used to decide when to exit the main loop in
    the AbstractOptimiser.optimise method, based on either time, iteration
    number, or error value """
    def __init__(self, t_lim=None, i_lim=None, e_lim=None):
        """ Initialise a Terminator object """
        self.t_lim = t_lim
        self.e_lim = e_lim
        self._num_iterations = i_lim
        self._init_timer()
    
    def set_initial_iteration(self, i):
        """ Use the initial iteration number to set the iteration number limit.
        This method is called in optimisers/abstract_optimiser.py, in the
        AbstractOptimiser.optimise method, before the main optimisation loop.
        """
        if self._num_iterations is not None:
            self.i_lim = i + self._num_iterations
        else:
            self.i_lim = None
    
    def ready_to_terminate(self, i=None, error=None):
        """
        Return True if ready to break out of the minimisation loop, otherwise
        return False.
        """
        if self.t_lim is not None:
            if self.time_elapsed() >= self.t_lim:
                return True
        
        if self.i_lim is not None:
            if i >= self.i_lim:
                return True
        
        if self.e_lim is not None:
            if error <= self.e_lim:
                return True
        
        return False

class DynamicTerminator(Terminator, _BatchGetter):
    """ Class for a terminator which terminates dynamically when the model is
    no longer providing consistent improvements in recently unseen points from
    the training set. If this class is used as the terminator for optimisation,
    then it must also be used as the batch-getter.

    TODO:
    -   Implement dynamic batch sizing based on probability of improvement on
        recently unseen points
    -   Investigate having separate smoothers for mean and standard deviation
        of the error, instead of one smoother for the probability of
        improvement
    -   Investigate smoothing the Boolean (in {-1, 1}) value of whether the
        batch was an improvement
    """
    def __init__(
        self,
        model,
        dataset,
        batch_size,
        replace=True,
        t_lim=None,
        i_lim=None,
        p_lim=0.001,
        i_interval=5,
        smooth_output=True,
        smooth_mean_reduction=True,
        smooth_std=True,
        smooth_n=50,
        smooth_x0=1,
    ):
        self.t_lim = t_lim
        self._p_lim = p_lim
        self._num_iterations = i_lim
        self._init_timer()

        if type(batch_size) is not int:
            raise TypeError("batch_size argument must be an integer")

        self.batch_size = batch_size
        self.replace = replace

        model.forward_prop(dataset.x_train)
        self._prev_mean_error = model.mean_error(dataset.y_train)
        self._model = model

        self._i_interval = i_interval
        self._i_next_update = 0
        self._i = 0
        self._p_improve_list = [smooth_x0]

        if smooth_output:
            self._output_smoother = MovingAverage(smooth_x0, smooth_n)
        else:
            self._output_smoother = None
        if smooth_mean_reduction:
            self._mean_reduction_smoother = MovingAverage(smooth_x0, smooth_n)
        else:
            self._mean_reduction_smoother = None
        if smooth_std:
            self._std_smoother = MovingAverage(smooth_x0, smooth_n)
        else:
            self._std_smoother = None

    def ready_to_terminate(self, i=None, error=None):
        """ Return True if ready to break out of the minimisation loop,
        otherwise return False """

        if self._p_lim is not None:
            if self._p_improve_list[-1] < self._p_lim:
                return True

        if self.t_lim is not None:
            if self.time_elapsed() >= self.t_lim:
                return True
        
        if self.i_lim is not None:
            if i >= self.i_lim:
                return True
        
        return False

    def get_batch(self, dataset):
        """ Get a batch for the next iteration, and also calculate the
        probability of improvement using the new batch, which will be used when
        the ready_to_terminate method is called """
        # Choose the batch
        batch_inds = np.random.choice(
            dataset.n_train,
            size=self.batch_size,
            replace=self.replace
        )
        x_batch = dataset.x_train[:, batch_inds]
        y_batch = dataset.y_train[:, batch_inds]

        if self._i >= self._i_next_update:
            # Calculate the probability of improvement
            self._model.forward_prop(x_batch)
            error = self._model.error(y_batch)
            mean_error = error.mean()
            mean_reduction = self._prev_mean_error - mean_error
            if self._mean_reduction_smoother is not None:
                mean_reduction = self._mean_reduction_smoother.smooth(
                    mean_reduction,
                )
            std_error = error.std()
            if self._std_smoother is not None:
                std_error = self._std_smoother.smooth(std_error)
            p_improve = mean_reduction / std_error
            if self._output_smoother is not None:
                p_improve = self._output_smoother.smooth(p_improve)
            
            self._p_improve_list.append(p_improve)
            self._prev_mean_error = mean_error
            self._i_next_update += self._i_interval
        
        self._i += 1

        # Return the inputs and outputs for the new batch
        return x_batch, y_batch

