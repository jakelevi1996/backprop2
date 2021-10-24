""" This module contains only the AbstractOptimiser class """

from optimisers.results import Result
from optimisers.evaluator import Evaluator
from optimisers.terminator import Terminator
from optimisers.batch import FullTrainingSet
from optimisers.columns import Iteration
from optimisers.timer import Timer

class AbstractOptimiser:
    """ This class is an abstract class for model-optimisation, with common
    training and evaluation logic which is used by specific optimisation
    algorithms (EG gradient descent, generalised Newton method, etc) which are
    implemented in separate modules. This model-optimisation class is applicable
    to all models and datasets in this repository """
    def __init__(self, line_search=None):
        """ Initialise this optimiser object, and store the optional line-search
        object which can be used by the optimise method. If line_search is None,
        then no line-search is used

        Inputs:
        -   line_search: (optional) object used to perform a line-search, which
            attempts to find an approximately locally optimal step-size to scale
            the change in parameters used for each iteration. Should be an
            instance of optimisers.LineSearch. Default is to not use a
            line-search
        """
        self.line_search = line_search

    def _get_step(self, model, x_batch, y_batch):
        """ This private method accepts a model and a batch of training data
        inputs and matching outputs, and returns the suggested change in
        model parameters and the vector of partial derivatives of the error
        function with respect to the parameters. It is assumed that this
        function propagates the inputs from the batch of training data forwards
        through the network, as well as calculating the gradient (which is
        needed if a line-search is used). This method should be overridden by
        each subclass.

        Inputs:
        -   model: model for which the suggested change in the parameters will
            be calculated. Should be an instance of models.NeuralNetwork
        -   x_batch: batch of input data used to calcualte the suggested change
            in the parameters. Should be a 2D numpy array, in which the 1st
            dimension matches the input dimension of the model, and the 2nd
            dimension matches the batch size used for this iteration
        -   y_batch: batch of output data used to calcualte the suggested change
            in the parameters. Should be a 2D numpy array, in which the 1st
            dimension matches the output dimension of the model, and the 2nd
            dimension matches the batch size used for this iteration

        Outputs:
        -   delta: suggested change in the parameters for the model, as a 1D
            numpy array
        -   dEdw: vector of partial derivatives of the error function with
            respect to the current model parameters, as a 1D numpy array
        """
        raise NotImplementedError()

    def optimise(
        self,
        model,
        dataset,
        evaluator=None,
        terminator=None,
        result=None,
        batch_getter=None,
        display_summary=True
    ):
        """ Optimise the given model to fit the given dataset. The code in this
        method is abstract, and depends on the output from the _get_step
        private method, which will be overriden by subclasses which are
        specific to different optimisation algorithms.

        Inputs:
        -   model: model that will be optimised. Should be an instance of
            models.NeuralNetwork
        -   dataset: dataset that model will be optimised to fit. Should be an
            instance of data.DataSet
        -   evaluator: (optional) object which is used to decide when to
            evaluate the model during optimisation, based on either time, or
            the number of iterations that have been completed. Should be an
            instance of optimisers.Evaluator. Default is to evaluate every 100
            iterations
        -   terminator: (optional) object which is used to decide when to stop
            optimising the model, based on either time, the number of
            iterations that have been completed, or the current value of the
            error function. Should be an instance of optimisers.Terminator.
            Default is to terminate after 1000 iterations
        -   result: (optional) result used to calculate, store, and display the
            progress of the model during optimisation. Can be configured with
            different columns from the optimisers.results.columns module. This
            object can also be passed to multiple different plotting functions.
            Should be an instance of optimisers.Result
        -   batch_getter: (optional) object which is used to choose batches
            from the training set used to optimise the model. Should be an
            instance of optimisers.batch._BatchGetter. Default is to use the
            full training set for each iteration of optimisation
        -   display_summary: (optional) whether or not to display a summary of
            the optimisation results after optimisation has finished. Should be
            a Boolean. Default is True
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
        i = result.get_iteration_number()
        evaluator.set_initial_iteration(i)
        terminator.set_initial_iteration(i)

        # Update the timer objects
        timer = Timer()
        evaluator.set_timer(timer)
        terminator.set_timer(timer)
        if not result.has_timer():
            result.set_timer(timer)

        # Begin the result and timer objects
        if not result.begun:
            result.begin()
        timer.begin()

        while True:
            # Evaluate the model
            if evaluator.ready_to_evaluate(i):
                result.update(model=model, dataset=dataset, iteration=i)

            # Get batch of training data
            x_batch, y_batch = batch_getter.get_batch(dataset.train)

            # Get gradient and initial step
            delta, dEdw = self._get_step(model, x_batch, y_batch)

            # Update parameters
            if self.line_search is not None:
                s = self.line_search.get_step_size(
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
