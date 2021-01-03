from time import perf_counter

class Terminator:
    """
    The Terminator class is used to decide when to exit the minimise function,
    based on either time, iteration number, or error value.

    TODO: add support for DBS
    """
    def __init__(self, t_lim=None, i_lim=None, e_lim=None):
        """ Initialise a Terminator object """
        self.t_lim = t_lim
        self.e_lim = e_lim
        self.i_lim_init = i_lim
    
    def begin(self, i):
        """ Reset the timer, and update the iteration limit depending on the
        starting iteration number """
        self.t_start = perf_counter()
        self.i_lim = i + self.i_lim_init
    
    def ready_to_terminate(self, i=None, error=None):
        """
        Return True if ready to break out of the minimisation loop, otherwise
        return False.
        """
        if self.t_lim is not None:
            if perf_counter() - self.t_start >= self.t_lim:
                return True
        
        if self.i_lim is not None:
            if i >= self.i_lim:
                return True
        
        if self.e_lim is not None:
            if error <= self.e_lim:
                return True
        
        return False
