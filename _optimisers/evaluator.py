from time import perf_counter

class Evaluator:
    """
    The Evaluator class is used to decide when to evaluate a model's
    performance during the minimise function, based on either time or iteration
    number.
    """
    def __init__(self, t_interval=None, i_interval=None):
        """ Initialise an Evaluator object """
        self.t_interval = t_interval
        self.i_interval = i_interval
        self.t_next_print = 0
    
    def begin(self, i):
        """ Reset the timer, and set the next iteration to evaluate """
        self.t_start = perf_counter()
        self.i_next_print = i
    
    def ready_to_evaluate(self, i):
        """
        Return True if ready to evaluate the model, otherwise return False.
        """
        if self.t_interval is not None:
            if perf_counter() - self.t_start >= self.t_next_print:
                self.t_next_print += self.t_interval
                return True
        
        if self.i_interval is not None:
            if i >= self.i_next_print:
                self.i_next_print += self.i_interval
                return True
        
        return False
