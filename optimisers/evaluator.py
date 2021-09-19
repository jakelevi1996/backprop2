from optimisers.timer import TimedObject

class Evaluator(TimedObject):
    """ The Evaluator class is used to decide when to evaluate a model's
    performance during the AbstractOptimiser.optimise method, based on either
    time or iteration number. """
    def __init__(self, t_interval=None, i_interval=None):
        """ Initialise an Evaluator object """
        self.t_interval = t_interval
        self.i_interval = i_interval
        self._init_timer()
    
    def set_initial_iteration(self, i):
        """ Use the initial iteration to set the next iteration to evaluate.
        This method is called in optimisers/abstract_optimiser.py, in the
        AbstractOptimiser.optimise method, before the main optimisation loop.

        If the current iteration is 0, then we will evaluate during iteration
        0, because we want to know the initial performance of the model, before
        optimisation starts. Otherwise we must be re-entering the optimise
        method, therefore we assume that the current iteration has been
        evaluated after the end of the last optimisation loop, so the next time
        we evaluate based on iteration number will be at the next multiple of
        self.i_interval (if this is not None) """
        if i == 0:
            self.i_next_print = 0
        elif self.i_interval is not None:
            self.i_next_print = i + self.i_interval
    
    def ready_to_evaluate(self, i):
        """ Return True if ready to evaluate the model, otherwise return False
        """
        if self.t_interval is not None:
            if self.time_elapsed() >= self.t_next_print:
                self.t_next_print += self.t_interval
                return True
        
        if self.i_interval is not None:
            if i >= self.i_next_print:
                self.i_next_print += self.i_interval
                return True
        
        return False

    def set_timer(self, timer):
        """ Set the timer for this object, and reset the t_next_print
        attribute, so that the ready_to_evaluate method works as expected when
        re-entering an optimisation loop with the same Evaluator object """
        self._timer = timer
        self.t_next_print = 0


class DoNotEvaluate(Evaluator):
    """ Class for an evaluator which never evaluates """
    def ready_to_evaluate(self, i):
        return False
