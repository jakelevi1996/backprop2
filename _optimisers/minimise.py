"""
Module to contain optimisation procedures and inner loop functions (EG HNGD,
backtracking, forward-tracking, etc.), agnostic to all models and objective
functions
"""
import numpy as np
from time import perf_counter
import models as m, data as d
from _optimisers.results import Result
from _optimisers.evaluator import Evaluator
from _optimisers.terminator import Terminator

def minimise(
    model,
    dataset,
    get_step,
    evaluator=None,
    terminator=None,
    line_search=None,
    result=None,
):
    """
    Abstract minimisation function, containing code which is common to all
    minimisation routines. Specific minimisation functions should call this
    function with a get_step callable, which should take a model and a dataset
    object, and return a step vector and a gradient vector.

    Inputs:
    -   ...

    TODO:
    -   Use batches
    -   Make this function private with a leading underscore
    -   Add `break_condition` method to Result class, and `while` loop here
        instead of `for` loop, so iteration can end based on one of several
        criteria EG iteration number, error function, time taken, DBS, etc.
        Breaking out of the optimisation loop could be handled by a Terminator
        class
    -   Evaluate the model every fixed time period, instead of every fixed
        iteration period? Could make this configurable with an input argument,
        and handled by an Evaluator class
    """
    if terminator is None:
        terminator = Terminator(i_lim=1000)
    if evaluator is None:
        evaluator = Evaluator(i_interval=100)
    if result is None:
        result = Result()
    if line_search is None:
        s = 1
    else:
        s = line_search.s

    # Set initial parameters and iteration counter
    w = model.get_parameter_vector()
    i = 0

    evaluator.begin()
    terminator.begin()
    result.begin()

    while True:
        # Evaluate the model
        if evaluator.ready_to_evaluate(i):
            result.update(model, dataset, i, s)

        # Get gradient and initial step
        delta, dEdw = get_step(model, dataset)
        
        # Update parameters
        if line_search is not None:
            s = line_search.get_step_size(
                model,
                dataset.x_train,
                dataset.y_train,
                w,
                delta,
                dEdw,
            )
            w += s * delta
        else:
            w += delta

        model.set_parameter_vector(w)

        i += 1
        
        # Check if ready to terminate minimisation
        if terminator.ready_to_terminate(i):
            break
        
    # Evaluate final performance
    result.update(model, dataset, i, s)
    if result.verbose:
        result.display_summary(i)

    return result
