"""
Module to contain optimisation procedures and inner loop functions (EG HNGD,
backtracking, forward-tracking, etc.), agnostic to all models and objective
functions
"""
from _optimisers.results import Result
from _optimisers.evaluator import Evaluator
from _optimisers.terminator import Terminator
from _optimisers.batch import FullTrainingSet
from _optimisers.columns import Iteration

def _minimise(
    model,
    dataset,
    get_step,
    evaluator=None,
    terminator=None,
    line_search=None,
    result=None,
    batch_getter=None,
    display_summary=True
):
    """ Abstract minimisation function, containing code which is common to all
    minimisation routines. Specific minimisation functions should call this
    function with a get_step callable, which should take a model and a dataset
    object, and return a step vector and a gradient vector.

    Inputs:
    -   ...
    -   get_step: callable which accepts the model and a batch of training data
        inputs and matching outputs, and returns delta (the suggested change in
        the parameters) and the vector of partial derivatives of the error
        function with respect to the parameters. It is assumed that this
        function propagates the inputs from the batch of training data forwards
        through the network, as well as calculating the gradient (which is
        needed if a line-search is used)
    -   ...
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
    if (
        result.has_column_type(Iteration)
        and len(result.get_values(Iteration)) > 0
    ):
        i = result.get_values(Iteration)[-1]
    else:
        i = 0

    if not result.begun:
        result.begin()
    evaluator.begin(i)
    terminator.begin(i)

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
    if display_summary and result.verbose:
        result.display_summary(i)

    return result
