"""
Module to contain optimisation procedures and inner loop functions (EG HNGD,
backtracking, forward-tracking, etc.), agnostic to all models and objective
functions

TODO: replace t and t0 (step size) with s and s0, and start_time with t0

TODO: make model evaluation and end of outer loop dependent on time, not on
iteration number
"""
import numpy as np
from time import perf_counter
import models as m, data as d
from results import Result

def backtrack_condition(s, E_new, E_0, delta_dot_dEdw, alpha):
    """
    Compares the actual reduction in the objective function to that which is
    expected from a first-order Taylor expansion. Returns True if a reduction in
    the step size is needed according to this criteria, otherwise returns False.

    TODO: try f_new > f_0 - (alpha * expected_reduction)
    """
    reduction = E_0 - E_new
    # if reduction == 0: return False
    expected_reduction = -s * delta_dot_dEdw
    return reduction < (alpha * expected_reduction)

def check_bad_step_size(s):
    if s == 0:
        print("s has converged to 0; resetting t0 s_old * beta ...")
        return True
    elif not np.isfinite(s):
        print("s has diverged; resetting t0 s_old/beta ...")
        return True
    else: return False

def line_search(model, x, y, w, s, delta, dEdw, alpha, beta, final_backstep):
    """
    ... TODO: use batches, and implement maximum number of steps of line search,
    using a for-loop with an `if condition: break` statement
    """
    # Calculate initial parameters
    E_0 = model.mean_error(y, x)
    E_old = E_0
    # s_old = s
    # w = model.get_parameter_vector()
    model.set_parameter_vector(w + s * delta)
    E_new = model.mean_error(y, x)
    
    delta_dot_dEdw = np.dot(delta, dEdw)
    bt_params = (E_0, delta_dot_dEdw, alpha)

    # Check initial backtrack condition
    if backtrack_condition(s, E_new, *bt_params):
        # Reduce step size until reduction is good enough and stops decreasing
        s *= beta
        E_old = E_new
        model.set_parameter_vector(w + s * delta)
        E_new = model.mean_error(y, x)

        # while backtrack_condition(s, E_new, *bt_params) or E_new < E_old:
        while backtrack_condition(s, E_new, *bt_params):
            s *= beta
            # if check_bad_step_size(s): return s_old * beta
            E_old = E_new
            model.set_parameter_vector(w + s * delta)
            E_new = model.mean_error(y, x)
        # if final_backstep or E_new > E_old: s /= beta
    else:
        # Track forwards until objective function stops decreasing
        s /= beta
        E_old = E_new
        model.set_parameter_vector(w + s * delta)
        E_new = model.mean_error(y, x)
        # while E_new < E_old:
        while not backtrack_condition(s, E_new, *bt_params):
            s /= beta
            # if check_bad_step_size(s): return s_old / beta
            E_old = E_new
            model.set_parameter_vector(w + s * delta)
            E_new = model.mean_error(y, x)
        # if final_backstep or E_new > E_old: s *= beta

    return s

def minimise(
    model, dataset, get_step, n_iters=1000, t_lim=5, E_lim=-np.inf,
    eval_every=100, line_search_flag=False, s0=1.0, alpha=0.5, beta=0.5,
    final_backstep=False, name=None, verbose=False
):
    """
    Abstract minimisation function, containing code which is common to all
    minimisation routines. Specific minimisation functions should call this
    function with a get_step callable, which should take a model and a dataset
    object, and return a step vector and a gradient vector.

    Inputs:
    -   ...

    TODO: use batches, and make this function private?
    """
    # Set initial parameters, step size and iteration counter
    w, s, i = model.get_parameter_vector(), s0, 0
    # Initialise result object, including start time of iteration
    result = Result(name, verbose)

    # while True:
    for i in range(n_iters):
        # Get gradient and initial step
        delta, dEdw = get_step(model, dataset)
        
        # # Check if delta is zero; if so, minimisation can't continue, so exit
        # if not np.any(delta): break

        # Evaluate the model
        if i % eval_every == 0: # TODO: make this condition time-based
            result.update(model, dataset, i, s)
        
        # Update parameters
        if line_search_flag:
            s = line_search(model, dataset.x_train, dataset.y_train, w, s,
                delta, dEdw, alpha, beta, final_backstep)
            w += s * delta
        else:
            w += delta
            model.set_parameter_vector(w)
        # model.set_parameter_vector(w)
        
        # # Increment loop counter and check loop condition
        # i += 1
        # if any([i >= n_iters,
        #     result.train_errors[-1] <= E_lim,
        #     result.time_elapsed() >= t_lim]): break

    # Evaluate final performance
    result.update(model, dataset, i, s)
    if verbose: result.display_summary(i)

    return result

def get_gradient_descent_step(model, dataset, learning_rate):
    """
    Method to get the descent step during each iteration of gradient-descent
    minimisation
    """
    dEdw = model.get_gradient_vector(dataset.x_train, dataset.y_train)

    return -learning_rate * dEdw, dEdw

def gradient_descent(
    model, dataset, learning_rate=1e-1, name="Gradient descent",
    final_backstep=False, **kwargs
):
    """ TODO: why is this ~10% slower than the old SGD function? """
    get_step = lambda model, dataset: get_gradient_descent_step(
        model, dataset, learning_rate)

    return minimise(model, dataset, get_step, name=name,
        final_backstep=final_backstep, **kwargs)



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

def generalised_newton(): raise NotImplementedError

def adam_optimiser(): raise NotImplementedError

def particle_swarm_optimiser(): raise NotImplementedError

def warmup(n_its=1000):
    """ Perform warmup routine """
    sin_data = d.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)
    n = m.NeuralNetwork(1, 1, [20])
    stochastic_gradient_descent(n, sin_data, n_its, n_its//10, verbose=True,
        name="Warmup")
