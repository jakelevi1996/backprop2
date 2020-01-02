"""
Module to contain optimisation procedures and inner loop functions (EG HNGD,
backtracking, forward-tracking, etc.), agnostic to all models and objective
functions
"""
import numpy as np
from time import perf_counter
import models as m, data as d, plotting as p, activations as a

def display_progress(i, t0, model, dataset):
    t = perf_counter() - t0
    e_train = model.mean_error(dataset.x_train, dataset.y_train)
    e_test = model.mean_error(dataset.x_test, dataset.y_test)
    progress_str = " | ".join([
        "Iter: {:7d}", "time: {:7.2f}s",
        "train error: {:.5f}", "test error: {:.5f}"
    ])
    print(progress_str.format(i, t, e_train, e_test))

def stochastic_gradient_descent(
    dataset, model, n_iters=4000, learning_rate=1e-3, print_every=50
):
    """
    stochastic_gradient_descent: perform simple stochastic gradient descent,
    given a dataset and a model

    Required inputs:
    -   dataset: should be an instance of data.DataSet, and should contain
        x_train, y_train, x_test, and y_test attributes
    -   model: should be an instance of models.NeuralNetwork, and should contain
        get_parameter_vector, get_gradient_vector, and set_parameter_vector
        methods
    """
    # Get initial parameters and start time
    w = model.get_parameter_vector()
    t0 = perf_counter()
    for i in range(n_iters):
        # Display progress
        if i % print_every == 0: display_progress(i, t0, model, dataset)
        # Update parameters
        dEdw = model.get_gradient_vector(dataset.x_train, dataset.y_train)
        w -= learning_rate * dEdw
        model.set_parameter_vector(w)
    # Print final error
    display_progress(n_iters, t0, model, dataset)
    print("Average time per iteration = {:.4f} ms".format(
        1e3 * (perf_counter() - t0) / n_iters
    ))

def linesearch_condition(): raise NotImplementedError

def sgd_forward_tracking(): raise NotImplementedError

def generalised_newton(): raise NotImplementedError

if __name__ == "__main__":
    # TODO: test training in functions in the training module; here just check
    # that optimisers work
    np.random.seed(0)
    # Test 1D:1D regression
    sin_data = d.SinusoidalDataSet1D1D(xlim=[0, 1], freq=0.5)
    n = m.NeuralNetwork(
        1, 1, [10, 10],
        # act_funcs=[a.Relu(), a.Identity()]
    )
    x_pred = np.linspace(-2, 3, 200).reshape(1, -1)
    y_pred = n(x_pred)
    p.plot_1D_regression(
        "Results/1D sin predictions (untrained)", sin_data, x_pred, y_pred,
    )
    stochastic_gradient_descent(
        sin_data, n, learning_rate=1e-4, n_iters=10000, print_every=1000
        # n_iters=20000, print_every=1000
    )
    y_pred = n(x_pred)
    p.plot_1D_regression(
        "Results/1D sin predictions (trained)", sin_data, x_pred, y_pred,
    )

    # Test 2D:4D regression (slow because of the massive data set; TODO: use
    # batching!!!)
    sin_data = d.SinusoidalDataSet2DnD(output_dim=4)
    n = m.NeuralNetwork(
        2, 4, [10],
    )
    x_pred = sin_data.x_test
    y_pred = n(x_pred)
    p.plot_2D_nD_regression(
        "Results/2Dto4D sin predictions (untrained)", 4, sin_data, y_pred
    )
    stochastic_gradient_descent(
        sin_data, n, learning_rate=1e-4, n_iters=100, print_every=10
        # n_iters=20000, print_every=1000
    )
    y_pred = n(x_pred)
    p.plot_2D_nD_regression(
        "Results/2Dto4D sin predictions (trained)", 4, sin_data, y_pred
    )
