from _optimisers.timer import TimedObject

class Terminator(TimedObject):
    """ The Terminator class is used to decide when to exit the main loop in the
    AbstractOptimiser.optimise method, based on either time, iteration number,
    or error value.

    TODO: add support for DBS
    """
    def __init__(self, t_lim=None, i_lim=None, e_lim=None):
        """ Initialise a Terminator object """
        self.t_lim = t_lim
        self.e_lim = e_lim
        self._num_iterations = i_lim
    
    def set_initial_iteration(self, i):
        """ Use the initial iteration number to set the iteration number limit.
        This method is called in _optimisers/abstract_optimiser.py, in the
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
