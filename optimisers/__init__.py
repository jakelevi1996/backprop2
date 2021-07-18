"""
Wrapper for modules, classes and functions related to optimisation, found in the
optimisers directory. See unit tests and scripts for usage examples.
"""
from optimisers.gradient_descent import gradient_descent, GradientDescent
from optimisers.gradient_descent import GradientDescentFixedLearningRate
from optimisers.generalised_newton import generalised_newton
from optimisers.evaluator import Evaluator, DoNotEvaluate
from optimisers.terminator import Terminator, DynamicTerminator
from optimisers.linesearch import LineSearch
from optimisers.results import Result
import data
import models

def warmup(n_its=1000):
    """
    Perform warmup routine; useful to call in scripts before testing the speed
    of an optimiser, because the process priority often appears to be initially
    slow
    """
    sin_data = data.Sinusoidal(1, 1, freq=1)
    n = models.NeuralNetwork(1, 1, [20])
    gradient_descent(
        n,
        sin_data,
        terminator=Terminator(i_lim=n_its),
        evaluator=Evaluator(i_interval=n_its//10),
        result=Result(name="Warmup", verbose=True),
        line_search=None
    )
