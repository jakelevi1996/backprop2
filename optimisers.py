"""
Module to contain optimisation procedures and inner loop functions (EG HNGD,
backtracking, forward-tracking, etc.), agnostic to all models and objective
functions
"""
import numpy as np
from time import perf_counter
import models as m, data as d

def get_train_test_errors(model, dataset):
    """
    get_train_test_errors: given a model and a dataset, return the mean (across
    data points) of the error in the training set and the test set, respectively
    """
    e_train = model.mean_error(dataset.y_train, dataset.x_train)
    e_test  = model.mean_error(dataset.y_test, dataset.x_test)
    return e_train, e_test

def eval_model(
    model, dataset, start_time, time_list, i, i_list,
    train_error_list, test_error_list, eval_print
):
    """
    eval_model: calculate the elapsed time, train error, and test error, update
    the time, iteration and error lists, and if eval_print == True then print
    the progress to stdout
    """
    # Calculate elapsed time, train error, and test error
    time_elapsed = perf_counter() - start_time
    e_train, e_test = get_train_test_errors(model, dataset)
    # Update the time, iteration and error lists
    time_list.append(time_elapsed)
    i_list.append(i)
    train_error_list.append(e_train)
    test_error_list.append(e_test)
    if eval_print:
        # Print progress to stdout
        progress_str = " | ".join([
            "Iter: {:6d}", "time: {:6.2f}s",
            "train error: {:8.5f}", "test error: {:8.5f}"
        ]).format(i, time_elapsed, e_train, e_test)
        print(progress_str)


def stochastic_gradient_descent(
    model, dataset, n_iters=4000, eval_every=500, eval_print=True,
    learning_rate=1e-3
):
    """
    stochastic_gradient_descent: given a model and a dataset, perform simple
    stochastic gradient descent to optimise the model for the dataset, using a
    fixed learning rate.

    Required inputs:
    -   dataset: should be an instance of data.DataSet, and should contain
        x_train, y_train, x_test, and y_test attributes
    -   model: should be an instance of models.NeuralNetwork, and should contain
        get_parameter_vector, get_gradient_vector, and set_parameter_vector
        methods
    """
    # Get initial parameters and start time
    w = model.get_parameter_vector()
    start_time = perf_counter()
    # Initialise error, time, and iteration lists
    train_error_list, test_error_list, time_list, i_list = [], [], [], []
    for i in range(n_iters):
        # Evaluate the model
        if i % eval_every == 0:
            eval_model(
                model, dataset, start_time, time_list, i, i_list,
                train_error_list, test_error_list, eval_print
            )
        # Update parameters
        dEdw = model.get_gradient_vector(dataset.x_train, dataset.y_train)
        w -= learning_rate * dEdw
        model.set_parameter_vector(w)
    # Evaluate final performance
    eval_model(
        model, dataset, start_time, time_list, n_iters, i_list,
        train_error_list, test_error_list, eval_print
    )
    # Print average iteration time
    print("Average time per iteration = {:.4f} ms".format(
        1e3 * (perf_counter() - start_time) / n_iters
    ))
    return train_error_list, test_error_list, time_list, i_list

def backtrack_condition(t, model, w, delta, dataset, alpha, dEdw, E0):
    """
    backtrack_condition: determine whether the current step size gives a
    sufficient reduction in the objective function; if the reduction is not good
    enough, then return True to indicate that the line search should back-track.

    The linesearch criterion is derived by rearranging a truncated first-order
    Taylor series (NB for gradient descent, <v, df/dx> should be negative):
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
    model, dataset, n_iters=5000, eval_every=500, eval_print=True,
    t0=1, alpha=0.8, beta=0.5
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
    -   eval_print: whether to print model performance to stdout every time it
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
    start_time = perf_counter()
    t = t0
    # Initialise error, time, iteration, and step-size lists
    train_error_list, test_error_list = [], []
    time_list, i_list, t_list = [], [], []
    for i in range(n_iters):
        # Evaluate the model
        if i % eval_every == 0:
            eval_model(
                model, dataset, start_time, time_list, i, i_list,
                train_error_list, test_error_list, eval_print
            )
            t_list.append(t)
        # Get the gradient and mean error for the current parameters
        dEdw = model.get_gradient_vector(dataset.x_train, dataset.y_train)
        E0 = model.mean_error(dataset.y_train)
        # Check if the current step size gives sufficient error reduction
        backtrack_params = (model, w, -dEdw, dataset, alpha, dEdw, E0)
        if backtrack_condition(t, *backtrack_params):
            # Reduce step size until error reduction is good enough
            t *= beta
            while backtrack_condition(t, *backtrack_params): t *= beta
        else:
            # Increase step size until error reduction is not good enough
            t /= beta
            while not backtrack_condition(t, *backtrack_params): t /= beta
            # Try also, keep forward tracking until E starts to increase

        w -= t * dEdw
        model.set_parameter_vector(w)
    # Evaluate final performance
    eval_model(
        model, dataset, start_time, time_list, n_iters, i_list,
        train_error_list, test_error_list, eval_print
    )
    # Print average iteration time
    print("Average time per iteration = {:.4f} ms".format(
        1e3 * (perf_counter() - start_time) / n_iters
    ))
    t_list.append(t)
    return train_error_list, test_error_list, time_list, i_list, t_list

def generalised_newton(): raise NotImplementedError

def adam_optimiser(): raise NotImplementedError

def particle_swarm_optimiser(): raise NotImplementedError

if __name__ == "__main__":
    np.random.seed(0)
    sin_data = d.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)
    n = m.NeuralNetwork(1, 1, [20])
    w = n.get_parameter_vector().copy()
    
    # stochastic_gradient_descent(n, sin_data, 100, 10)
    # n.set_parameter_vector(w)
    # sgd_2way_tracking(n, sin_data, 100, 10)
    
    stochastic_gradient_descent(n, sin_data, 10000, 1000)
    n.set_parameter_vector(w)
    sgd_2way_tracking(n, sin_data, 10000, 1000)
