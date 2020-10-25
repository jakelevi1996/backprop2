"""
Module to contain optimisation procedures and inner loop functions (EG HNGD,
backtracking, forward-tracking, etc.), agnostic to all models and objective
functions
"""
from _optimisers.results import Result
from _optimisers.evaluator import Evaluator
from _optimisers.terminator import Terminator
from _optimisers.batch import FullTrainingSet

def minimise(
    model,
    dataset,
    get_step,
    evaluator=None,
    terminator=None,
    line_search=None,
    result=None,
    batch_getter=None
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
    if batch_getter is None:
        batch_getter = FullTrainingSet()

    # Set initial parameters and iteration counter
    w = model.get_parameter_vector()
    i = 0

    result.begin()
    evaluator.begin()
    terminator.begin()

    while True:
        # Evaluate the model
        if evaluator.ready_to_evaluate(i):
            result.update(model=model, dataset=dataset, iteration=i)
        
        # Get batch of training data
        x_batch, y_batch = batch_getter.get_batch(dataset)

        # Get gradient and initial step
        delta, dEdw = get_step(model, x_batch, y_batch)
        
        # Update parameters
        if line_search is not None:
            s = line_search.get_step_size(
                model,
                x_batch,
                y_batch,
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
    result.update(model=model, dataset=dataset, iteration=i)
    if result.verbose:
        result.display_summary(i)

    return result
