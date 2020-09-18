"""
Wrapper for modules, classes and functions related to optimisation, found in the
_optimisers directory. See unit tests and scripts for usage examples.
"""
from _optimisers.optimisers import gradient_descent, generalised_newton
from _optimisers.evaluator import Evaluator
from _optimisers.terminator import Terminator
from _optimisers import results
from _optimisers.results import Result
