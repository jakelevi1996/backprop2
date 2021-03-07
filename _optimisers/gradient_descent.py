""" This module contains the GradientDescentFixedLearningRate and
GradientDescent classes, and the gradient_descent wrapper function. """

from _optimisers.abstract_optimiser import AbstractOptimiser
from _optimisers.results import Result

class GradientDescentFixedLearningRate(AbstractOptimiser):
    """ Class to represent a gradient-descent optimisation algorithm with a
    fixed learning rate and no line-search. Instead of using this class
    directly, in general it is recommended to use the gradient_descent wrapper
    function instead, with the line_search keyword-argument left as the default
    value of None """
    def __init__(self, learning_rate):
        """ Initialise this GradientDescentFixedLearningRate object, setting the
        _learning_rate private attribute to use in the _get_step method, and
        calling the parent initialiser to set the line_search attribute to None
        (so that no line-search is used during optimisation) """
        self._learning_rate = learning_rate
        super().__init__(line_search=None)

    def _get_step(self, model, x_batch, y_batch):
        dEdw = model.get_gradient_vector(x_batch, y_batch)
        delta = -self._learning_rate * dEdw

        return delta, dEdw

class GradientDescent(AbstractOptimiser):
    """ Class to represent a gradient-descent optimisation algorithm with a
    variable learning rate, set by a LineSearch object. Instead of using this
    class directly, in general it is recommended to use the gradient_descent
    wrapper function instead, with the line_search keyword-argument provided
    with an instance of optimisers.LineSearch. """
    def _get_step(self, model, x_batch, y_batch):
        dEdw = model.get_gradient_vector(x_batch, y_batch)
        delta = -dEdw

        return delta, dEdw

def gradient_descent(
    model,
    dataset,
    learning_rate=1e-1,
    result=None,
    line_search=None,
    **kwargs
):
    """ Perform gradient descent to optimise the given model to fit the given
    dataset. This function wraps the optimise method of the 2 classes in this
    module. In general it is recommended to use this wrapper function instead of
    those classes.

    Inputs:
    -   model: model that will be optimised. Should be an instance of
        models.NeuralNetwork
    -   dataset: dataset that model will be optimised to fit. Should be an
        instance of data.DataSet
    -   learning_rate: (optional) learning rate that is used for
        gradient-descent optimisation, if no line-search object is passed to
        this function. Should be a positive float. If line_search is not None,
        then this argument is ignored
    -   result: (optional) result used to calculate, store, and display the
        progress of the model during optimisation. Should be an instance of
        optimisers.Result
    -   line_search: (optional) object used to perform a line-search, which
        attempts to find an approximately locally optimal step-size to scale the
        change in parameters used for each iteration. Should be an instance of
        optimisers.LineSearch. Default is to not use a line-search
    -   **kwargs: (optional) extra keyword arguments that are passed through to
        the AbstractOptimiser.optimise method, such as objects used to decide
        when to evaluate the model, terminate optimisation, and choose batches
        from the training set
    """
    if line_search is None:
        optimiser = GradientDescentFixedLearningRate(learning_rate)
    else:
        optimiser = GradientDescent(line_search)
    
    if result is None:
        result = Result("Gradient descent")

    result = optimiser.optimise(
        model,
        dataset,
        result=result,
        **kwargs
    )

    return result
