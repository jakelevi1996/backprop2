"""
Module to contain optimisation procedures and inner loop functions (EG HNGD,
backtracking, forward-tracking, etc.), agnostic to all models and objective
functions

TODO: make model evaluation and end of outer loop dependent on time, not on
iteration number?
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

def line_search(model, x, y, w, s, delta, dEdw, alpha, beta):
    """
    ... TODO: use batches, and implement maximum number of steps of line search,
    using a for-loop with an `if condition: break` statement
    """
    # Calculate initial parameters
    E_0 = model.mean_error(y, x)
    E_old = E_0
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

        while backtrack_condition(s, E_new, *bt_params) or E_new < E_old:
            s *= beta
            E_old = E_new
            model.set_parameter_vector(w + s * delta)
            E_new = model.mean_error(y, x)
        if E_new > E_old:
            s /= beta
    else:
        # Track forwards until objective function stops decreasing
        s /= beta
        E_old = E_new
        model.set_parameter_vector(w + s * delta)
        E_new = model.mean_error(y, x)
        while E_new < E_old:
            s /= beta
            E_old = E_new
            model.set_parameter_vector(w + s * delta)
            E_new = model.mean_error(y, x)
        if E_new > E_old:
            s *= beta

    return s

def minimise(
    model,
    dataset,
    get_step,
    n_iters=1000,
    t_lim=5,
    E_lim=-np.inf,
    eval_every=100,
    line_search_flag=False,
    s0=1.0,
    alpha=0.5,
    beta=0.5,
    name=None,
    verbose=False,
    result_file=None
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
    # Set initial parameters, step size and iteration counter
    w = model.get_parameter_vector()
    s = s0
    i = 0
    next_eval = 0
    # Initialise result object, including start time of iteration
    result = Result(name, verbose, result_file)

    # while True:
    for i in range(n_iters):
        # Get gradient and initial step
        delta, dEdw = get_step(model, dataset)

        # Evaluate the model
        if i >= next_eval:
            result.update(model, dataset, i, s)
            next_eval += eval_every
        
        # Update parameters
        if line_search_flag:
            s = line_search(
                model,
                dataset.x_train,
                dataset.y_train,
                w,
                s,
                delta,
                dEdw,
                alpha,
                beta,
            )
            w += s * delta
        else:
            w += delta

        model.set_parameter_vector(w)
        
    # Evaluate final performance
    result.update(model, dataset, n_iters, s)
    if verbose:
        result.display_summary(n_iters)

    return result

def get_gradient_descent_step(model, dataset, learning_rate):
    """
    Method to get the descent step during each iteration of gradient-descent
    minimisation
    """
    dEdw = model.get_gradient_vector(dataset.x_train, dataset.y_train)

    return -learning_rate * dEdw, dEdw

def gradient_descent(
    model,
    dataset,
    learning_rate=1e-1,
    name="Gradient descent",
    **kwargs
):
    """ TODO: why is this ~10% slower than the old SGD function? """
    get_step = lambda model, dataset: get_gradient_descent_step(
        model, dataset, learning_rate)

    result = minimise(model, dataset, get_step, name=name, **kwargs)

    return result

class NewtonStepCalculator():
    # TODO: make it possible to specify that the block indices should be
    # regenerated on every iteration
    def __init__(self, model, max_block_size, max_step, learning_rate):
        self.model = model

        # Get random indices for block-diagonalisation of weights in each layer
        self.weight_inds = [
            np.array_split(
                np.random.permutation(layer.num_weights),
                np.ceil(layer.num_weights / max_block_size)
            ) for layer in model.layers
        ]

        # Get random indices for block-diagonalisation of biases in each layer
        self.bias_inds = [
            np.array_split(
                np.random.permutation(layer.num_bias),
                np.ceil(layer.num_bias / max_block_size)
            ) for layer in model.layers
        ]

        self.delta = np.empty(model.num_params)

        self.max_step = max_step
        self.learning_rate = learning_rate
    
    def get_step(self, model, dataset):
        # Get gradient vector
        dEdw = model.get_gradient_vector(dataset.x_train, dataset.y_train)
        # Get Hessian blocks
        (hess_block_list, hess_inds_list) = model.get_hessian_blocks(
            dataset.x_train, dataset.y_train, self.weight_inds, self.bias_inds
        )
        # Iterate through each Hessian block
        for hess_block, hess_inds in zip(hess_block_list, hess_inds_list):
            # Rotate gradient into eigenbasis of Hessian
            evals, evecs = np.linalg.eigh(hess_block)
            grad_rot = np.matmul(evecs.T, dEdw[hess_inds])
            # Take a Newton step in directions in which this step is not too big
            step_rot = np.where(
                (self.max_step * np.abs(evals)) > np.abs(grad_rot),
                -grad_rot / np.abs(evals),
                -self.learning_rate * grad_rot
            )
            # Rotate gradient back into original coordinate system and return
            self.delta[hess_inds] = np.matmul(evecs, step_rot)
        
        return self.delta, dEdw


def generalised_newton(
    model,
    dataset,
    learning_rate=1e-1,
    max_block_size=7,
    max_step=1,
    name="Gradient descent",
    **kwargs
):
    newton_step_calculator = NewtonStepCalculator(
        model,
        max_block_size,
        max_step,
        learning_rate
    )

    get_step = lambda model, dataset: newton_step_calculator.get_step(
        model,
        dataset
    )

    result = minimise(model, dataset, get_step, name=name, **kwargs)

    return result

def adam_optimiser(): raise NotImplementedError

def particle_swarm_optimiser(): raise NotImplementedError

def warmup(n_its=1000):
    """
    Perform warmup routine; useful to call in scripts before testing the speed
    of an optimiser, because the process priority often appears to be initially
    slow
    """
    sin_data = d.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)
    n = m.NeuralNetwork(1, 1, [20])
    gradient_descent(
        n,
        sin_data,
        n_iters=n_its,
        eval_every=n_its//10,
        verbose=True,
        name="Warmup",
        line_search_flag=False
    )
