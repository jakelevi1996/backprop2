"""
Wrapper for modules, classes and functions related to optimisation, found in the
_optimisers directory. See unit tests and scripts for usage examples.
"""
from _optimisers.gradient_descent import gradient_descent
from _optimisers.generalised_newton import generalised_newton
from _optimisers.evaluator import Evaluator
from _optimisers.terminator import Terminator
from _optimisers import results
from _optimisers.results import Result
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
        n_iters=n_its,
        eval_every=n_its//10,
        verbose=True,
        name="Warmup",
        line_search_flag=False
    )
