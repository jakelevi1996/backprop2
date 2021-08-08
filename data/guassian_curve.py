from data.regression import Regression

class SumOfGaussianCurves(Regression):
    pass

class GaussianCurve(Regression):
    def __init__(self, input_loc=None, input_scale=None):
        """ Initialise a GaussianCurve object """
