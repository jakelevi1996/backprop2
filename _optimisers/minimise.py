""" This module contains an abstract function for model-optimisation, with
common training and evaluation logic which is used by specific optimisation
algorithms (EG gradient descent, generalised Newton method) in different
modules. This model-optimisation function is applicable to all models and
datasets in this repository """
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
    """ Abstract model-optimisation function, containing common training and
    evaluation logic which is used by specific optimisation algorithms in
    different modules. Specific minimisation functions should call this function
    with a get_step callable, which should take a model and a dataset object,
    and return a step vector and a gradient vector.

    Inputs:
    -   model: model that will be optimised. Should be an instance of
        models.NeuralNetwork
    -   dataset: dataset that model will be optimised to fit. Should be an
        instance of data.DataSet
    -   get_step: callable which accepts the model and a batch of training data
        inputs and matching outputs, and returns delta (the suggested change in
        the parameters) and the vector of partial derivatives of the error
        function with respect to the parameters. It is assumed that this
        function propagates the inputs from the batch of training data forwards
        through the network, as well as calculating the gradient (which is
        needed if a line-search is used)
    -   evaluator: (optional) object which is used to decide when to evaluate
        the model during optimisation, based on either time, or the number of
        iterations that have been completed. Should be an instance of
        optimisers.Evaluator. Default is to evaluate every 100 iterations
    -   terminator: (optional) object which is used to decide when to stop
        optimising the model, based on either time, the number of iterations
        that have been completed, or the current value of the error function.
        Should be an instance of optimisers.Terminator. Default is to terminate
        after 1000 iterations
    -   line_search: (optional) object used to perform a line-search, which
        attempts to find an approximately locally optimal step-size to scale the
        change in parameters used for each iteration. Should be an instance of
        optimisers.LineSearch. Default is to not use a line-search
    -   result: (optional) result used to calculate, store, and display the
        progress of the model during optimisation. Can be configured with
        different columns from the optimisers.results.columns module. This
        object can also be passed to multiple different plotting functions.
        Should be an instance of optimisers.Result
    -   batch_getter: (optional) object which is used to choose batches from the
        training set used to optimise the model. Should be an instance of
        optimisers.batch._BatchGetter. Default is to use the full training set
        for each iteration of optimisation
    -   display_summary: (optional) whether or not to display a summary of the
        optimisation results after optimisation has finished. Should be a
        Boolean. Default is True
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
