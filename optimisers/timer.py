""" This module contains the Timer and TimedObject classes """

from time import perf_counter

class Timer:
    """ Timer object, which can be shared between different classes, for
    example Result, Evaluator, Terminator, which all require access to a shared
    timer
    """
    def __init__(self):
        """ Initialise a Timer object """
        self._start_time = None

    def begin(self):
        """ Begin the Timer object by recording the current time """
        self._start_time = perf_counter()
    
    def time_elapsed(self):
        """ Return the time elapsed in seconds since the begin method of this
        Timer obejct was called. The time elapsed is returned as a float. The
        Timer.begin method of this timer object must have been called before
        this method is called """
        return perf_counter() - self._start_time

class TimedObject:
    """ Class representing an object which has a timer. This timer can be
    shared between multiple objects (EG a Result, an Evaluator, and a
    Terminator). This class is intended to be inherited from, but not
    instantiated directly """
    
    def _init_timer(self):
        """ Initialise a _timer attrinute for this TimedObject object. This
        method MUST be called by all subclasses of TimedObject when they are
        initialised """
        self._timer = None

    def set_timer(self, timer):
        """ Set the timer for this object """
        self._timer = timer
    
    def has_timer(self):
        """ Check if this object has been set with a valid timer object """
        return isinstance(self._timer, Timer)
    
    def time_elapsed(self):
        """ Return the time elapsed in seconds since the begin method of this
        obejct's timer was called. The time elapsed is returned as a float. The
        set_timer method of this object and the Timer.begin method of this
        object's timer must have been called before this method is called """
        return self._timer.time_elapsed()
