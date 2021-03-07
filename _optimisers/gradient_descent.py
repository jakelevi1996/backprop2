from _optimisers.minimise import AbstractOptimiser
from _optimisers.results import Result

class GradientDescentFixedLearningRate(AbstractOptimiser):
    def __init__(self, learning_rate):
        self._learning_rate = learning_rate
        super().__init__(line_search=None)

    def _get_step(self, model, x_batch, y_batch):
        """ Method to get the descent step during each iteration of
        gradient-descent minimisation, using a learning rate """
        dEdw = model.get_gradient_vector(x_batch, y_batch)
        delta = -self._learning_rate * dEdw

        return delta, dEdw

class GradientDescent(AbstractOptimiser):
    def _get_step(self, model, x_batch, y_batch):
        """ Method to get the descent step during each iteration of
        gradient-descent minimisation, without a learning rate (this is intended
        to be used with a line-search) """
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
    """ TODO: why is this ~10% slower than the old SGD function? """
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
