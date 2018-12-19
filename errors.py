import numpy as np
import matplotlib.pyplot as plt

class ErrorFunction():
    name = None
    def __call__(self, x): raise NotImplementedError
    def dEdy(self, y, target): raise NotImplementedError

class SumOfSquares(ErrorFunction):
    # Should account for higher dimensions/multiple data points?
    def __call__(self, y, target): return 0.5 * np.square(y - target).sum()
    def dEdy(self, y, target): return (y - target).sum()

class BinaryCrossEntropy(ErrorFunction):
    def __call__(self, y, target):
        return - (target * np.log(y) + (1.0 - target) * np.log(1.0 - y)).sum()

    def dEdy(self, y, target):
        return ((y - target) / (y * (1.0 - y))).sum()
