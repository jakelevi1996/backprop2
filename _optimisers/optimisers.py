"""
Module to contain optimisation procedures and inner loop functions (EG HNGD,
backtracking, forward-tracking, etc.), agnostic to all models and objective
functions
"""
import numpy as np
from time import perf_counter
import models as m, data as d
from _optimisers.linesearch import line_search
from _optimisers.results import Result
from _optimisers.evaluator import Evaluator
from _optimisers.terminator import Terminator

def minimise(
    model,
    dataset,
    get_step,
    evaluator=None,
    terminator=None,
    line_search_flag=False,
    s0=1.0,
    alpha=0.5,
    beta=0.5,
    name=None,
    verbose=False,
    result_file=None
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

    # Set initial parameters, step size and iteration counter
    w = model.get_parameter_vector()
    s = s0
    i = 0
    # Initialise result object, including start time of iteration
    result = Result(name, verbose, result_file)

    evaluator.begin()
    terminator.begin()

    while True:
        # Get gradient and initial step
        delta, dEdw = get_step(model, dataset)

        # Evaluate the model
        if evaluator.ready_to_evaluate(i):
            result.update(model, dataset, i, s)
        
        # Update parameters
        if line_search_flag:
            s = line_search(
                model,
                dataset.x_train,
                dataset.y_train,
                w,
                s,
                delta,
                dEdw,
                alpha,
                beta,
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
    if verbose:
        result.display_summary(i)

    return result
