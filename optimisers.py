"""
Wrapper for modules, classes and functions related to optimisation, found in the
_optimisers directory. See unit tests and scripts for usage examples.
"""
from _optimisers.gradient_descent import gradient_descent
from _optimisers.generalised_newton import generalised_newton
from _optimisers.evaluator import Evaluator
from _optimisers.terminator import Terminator
from _optimisers.linesearch import LineSearch
from _optimisers import results
from _optimisers.results import Result
from _optimisers import batch
import data, models

def warmup(n_its=1000):
    """
    Perform warmup routine; useful to call in scripts before testing the speed
    of an optimiser, because the process priority often appears to be initially
    slow
    """
    sin_data = data.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)
    n = models.NeuralNetwork(1, 1, [20])
    gradient_descent(
        n,
        sin_data,
        terminator=Terminator(i_lim=n_its),
        evaluator=Evaluator(i_interval=n_its//10),
        result=Result(name="Warmup", verbose=True),
        line_search=None
    )
