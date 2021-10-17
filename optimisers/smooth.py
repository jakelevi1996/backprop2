""" Module containing different classes for smoothing an input signal """

import numpy as np

class _Smoother:
    """ Abstract parent class for smoothers """
    def __init__(self, x0):
        """ Initialise a smoother object, using x0 as the initial value of x,
        to initialise this smoother object """
        raise NotImplementedError

    def smooth(self, x):
        """ Given a noisy input value x, return a smoothed estimate of x,
        calculated using the current version of x, and previous values of x
        from previous calls to this method """
        raise NotImplementedError

class Identity(_Smoother):
    def __init__(self, x0):
        """ Initialise a trivial smoother which doesn't do any actual
        smoothing, and just returns the input x as the smoothed output """

    def smooth(self, x):
        return x

class Exponential(_Smoother):
    def __init__(self, x0, alpha=0.25):
        """ Initialise an exponential smoother, which smooths a noisy signal x
        according to the following equation:

            y(n) = alpha*x(n) + (1 - alpha)*y(n-1)

        Inputs:
        -   x0: initial value of x, used as the initial state (output value)
            for this exponential smoother
        -   alpha: constant in (0, 1) that determines how much weight should be
            given to new input values. If alpha is large, then more weight is
            given to new input values, meaning the output is less smooth, and
            reacts more quickly to changes in x. A value of alpha = 0.1 means
            that for each new input value, 10% of the weight of the
            corresponding output value comes from the new input value, and 90%
            comes from previous input values, with more recent input values
            contributing exponentially more to the output value than older
            input values
        """
        assert (alpha > 0) and (alpha < 1), "alpha must be in (0, 1)"
        self._alpha = alpha
        self._y = x0

    def smooth(self, x):
        self._y = (self._alpha * x) + ((1.0 - self._alpha) * self._y)
        return self._y

class MovingAverage(_Smoother):
    def __init__(self, x0, n=5):
        """ Initialise a moving average smoother, which smooths a noisy signal
        x according to the following equation:

            y(m) = (x(m) + x(m-1) + ... + x(m-n+1)) / n

        Inputs:
        -   x0: initial value of x, also used as assumed values of x for m < 0
        -   n: number of most recent input values to average in order to
            calculate the smoothed output value
        """
        self._buffer = np.full(n, x0, dtype=float)
        self._i = 0
        self._n = n
        self._x0 = x0

    def smooth(self, x):
        self._buffer[self._i] = x
        self._i += 1
        if self._i == self._n:
            self._i = 0
        y = self._buffer.mean()
        return y

    def reset(self):
        self._buffer[:] = self._x0

class MovingMaximum(_Smoother):
    def __init__(self, x0, n=5):
        """ Initialise a moving maximum smoother, which smooths a noisy signal
        x according to the following equation:

            y(m) = max(x(m), x(m-1), ... , x(m-n+1))

        Inputs:
        -   x0: initial value of x, also used as assumed values of x for m < 0
        -   n: number of most recent input values to store in order to
            calculate the smoothed output value
        """
        self._buffer = [x0] * n
        self._i = 0
        self._n = n

    def smooth(self, x):
        self._buffer[self._i] = x
        self._i += 1
        if self._i == self._n:
            self._i = 0
        y = max(self._buffer)
        return y

class KalmanFilter(_Smoother):
    def __init__(self):
        raise NotImplementedError

smoother_dict = {
    s.__name__: s
    for s in [
        Identity,
        Exponential,
        MovingAverage,
        MovingMaximum,
    ]
}
