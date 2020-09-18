"""
This module contains old optimiser code. The only reason it still exists is to
debug why the new optimiser code is sometimes slower than the old optimiser code
(there is/will be a script in the Scripts directory to explore why this is the
case). Once this has been determined and debugged, this module should be
deleted.
"""
import numpy as np
from results import Result

def stochastic_gradient_descent(
    model, dataset, n_iters=4000, eval_every=500, verbose=True,
    learning_rate=1e-3, name="SGD"
):
    """
    stochastic_gradient_descent: given a model and a dataset, perform simple
    stochastic gradient descent to optimise the model for the dataset, using a
    fixed learning rate.

    Required inputs:
    -   model: should be an instance of models.NeuralNetwork, and should contain
        get_parameter_vector, get_gradient_vector, and set_parameter_vector
        methods
    -   dataset: should be an instance of data.DataSet, and should contain
        x_train, y_train, x_test, and y_test attributes
    """
    # Get initial parameters and start time, and initialise results dictionary
    w = model.get_parameter_vector()
    result = Result(name)
    for i in range(n_iters):
        # Evaluate the model
        if i % eval_every == 0: result.update(model, dataset, i, 1)
        # Update parameters
        dEdw = model.get_gradient_vector(dataset.x_train, dataset.y_train)
        w -= learning_rate * dEdw
        model.set_parameter_vector(w)
    # Evaluate final performance
    result.update(model, dataset, n_iters, 1)
    if verbose: result.display_summary(n_iters)
    return result

def old_backtrack_condition(t, model, w, delta, dataset, alpha, dEdw, E0):
    """
    backtrack_condition: determine whether the current step size gives a
    sufficient reduction in the objective function; if the reduction is not good
    enough, then return True to indicate that the line search should back-track.

    The linesearch criterion is derived by rearranging a truncated first-order
    Taylor series in terms of the reduction in the objective function (NB for
    gradient descent, <v, df/dx> should be negative):
    *   f(x + t*v) = f(x) + t * <v, df/dx> + ...
    *   => f(x) - f(x + t*v) ~~ - t * <v, df/dx>

    IE to a first order approximation, the reduction in the objective function
    from taking a step described by the vector t*v should be equal to
    -t*<v,df/dx>. When the curvature is positive, as t->0, the actual reduction
    in the step size will approach but never reach this value, so the threshold
    for the minimum reduction in the objective function is scaled by a constant
    alpha.
    """
    model.set_parameter_vector(w + t * delta)
    E_new = model.mean_error(dataset.y_train, dataset.x_train)
    min_reduction = -alpha * t * np.dot(delta, dEdw)
    return min_reduction > (E0 - E_new)

def sgd_2way_tracking(
    model, dataset, n_iters=5000, eval_every=500, verbose=True,
    t0=1, alpha=0.8, beta=0.5, name="SGD with line-search"
):
    """
    sgd_2way_tracking: given a model and a dataset, perform stochastic gradient
    descent to optimise the model for the dataset, using a bidirectional
    line-search to find a good step size during each iteration; the step size
    which is found during each iteration persists as the initial step size
    during the next iteration.

    Inputs:
    -   model: the model which will be optimised
    -   dataset: the dataset which the model will be trained on
    -   n_iters: the number of outer loop iterations to perform
    -   eval_every: how frequently to evaluate model performance
    -   verbose: whether to print model performance to stdout every time it
        is evaluated
    -   t0: initial step size to take
    -   alpha: fraction of the theoretical approximate step size which is
        considered acceptible
    -   beta: factor with which the step size will be multiplied during each
        iteration of back-tracking; for forward-tracking, it is the inverse of
        the factor by which the step-size is multiplied
    """
    # Get initial parameters, step size and start time
    w = model.get_parameter_vector()
    t = t0
    result = Result(name)
    for i in range(n_iters):
        # Evaluate the model
        if i % eval_every == 0:
            result.update(model, dataset, i, t)
        # Get the gradient and mean error for the current parameters
        dEdw = model.get_gradient_vector(dataset.x_train, dataset.y_train)
        E0 = model.mean_error(dataset.y_train)
        # Check if the current step size gives sufficient error reduction
        backtrack_params = (model, w, -dEdw, dataset, alpha, dEdw, E0)
        if old_backtrack_condition(t, *backtrack_params):
            # Reduce step size until error reduction is good enough
            t *= beta
            while old_backtrack_condition(t, *backtrack_params): t *= beta
        else:
            # Increase step size until error reduction is not good enough
            t /= beta
            while not old_backtrack_condition(t, *backtrack_params): t /= beta
            # Try also, keep forward tracking until E starts to increase

        w -= t * dEdw
        model.set_parameter_vector(w)
    # Evaluate final performance
    result.update(model, dataset, n_iters, t)
    if verbose: result.display_summary(n_iters)
    return result
