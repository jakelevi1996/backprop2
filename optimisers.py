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

class Result():
    """
    Class to store the results of optimisation in a single object which can be
    passed directly to plotting and/or analysis functions. Also contains methods
    for updating and displaying results

    TODO: Make this class configurable, so columns such as step-size and |x| are
    optional, and the column width and format spec for each column is
    configurable. Also implement saving and loading of results
    """
    def __init__(self, name=None, verbose=True):
        """
        Store the name of the experiment (which is useful later when displaying
        results), display table headers, initialise lists for objective function
        evaluations and the time and iteration number for each evaluation, and
        record the start time for the results list
        """
        self.name = name if (name is not None) else "Unnamed experiment"
        if verbose: self.display_headers()
        self.verbose = verbose

        self.train_errors   = []
        self.test_errors    = []
        self.times          = []
        self.iters          = []
        self.step_size      = []
        self.start_time     = perf_counter()
        # TODO: DBS criterion
    
    def update(self, model, dataset, i, s):
        t = perf_counter() - self.start_time
        e_train = model.mean_error(dataset.y_train, dataset.x_train)
        e_test  = model.mean_error(dataset.y_test, dataset.x_test)
        self.train_errors.append(e_train)
        self.test_errors.append(e_test)
        self.times.append(t)
        self.iters.append(i)
        self.step_size.append(s)
        if self.verbose: self.display_last()
    
    def display_headers(self):
        # num_fields, field_width = 3, 10
        print("\nPerforming test \"{}\"...".format(self.name))
        print("{:9} | {:8} | {:11} | {:11} | {:10}".format(
            "Iteration", "Time (s)", "Train error", "Test error", "Step size"))
        print(" | ".join("-" * i for i in [9, 8, 11, 11, 10]))

    def display_last(self):
        print("{:9d} | {:8.3f} | {:11.5f} | {:11.5f} | {:10.4f}".format(
            self.iters[-1], self.times[-1], self.train_errors[-1],
            self.test_errors[-1], self.step_size[-1]))

    def display_summary(self, n_iters):
        t_total = perf_counter() - self.start_time
        t_mean = t_total / n_iters
        print("-" * 50,
            "{:30} = {}".format("Test name", self.name),
            "{:30} = {:,.4f} s".format("Total time", t_total),
            "{:30} = {:,}".format("Total iterations", n_iters),
            "{:30} = {:.4f} ms".format("Average time per iteration",
                1e3 * t_mean),
            "{:30} = {:,.1f}".format("Average iterations per second",
                1 / t_mean),
            sep="\n", end="\n\n")
    
    def save(self, filename): raise NotImplementedError
    def load(self, filename): raise NotImplementedError


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

def backtrack_condition(t, model, w, delta, dataset, alpha, dEdw, E0):
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
    result.update(model, dataset, n_iters, 1)
    if verbose: result.display_summary(n_iters)
    return result

def generalised_newton(): raise NotImplementedError

def adam_optimiser(): raise NotImplementedError

def particle_swarm_optimiser(): raise NotImplementedError

def warmup(n_its=1000):
    """ Perform warmup routine """
    np.random.seed(0)
    sin_data = d.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)
    n = m.NeuralNetwork(1, 1, [20])
    stochastic_gradient_descent(n, sin_data, n_its, n_its//10, verbose=True,
        name="Warmup")

if __name__ == "__main__":
    warmup()
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
