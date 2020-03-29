"""
Module to contain network training experiments.

TODO: some of these experiments could probably be removed.
"""

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

import models as m, data as d, optimisers as o, plotting as p, \
    activations as a, errors as e

def save_results_dict(): raise NotImplementedError

def load_results_dict(): raise NotImplementedError

def train_1D_regression_sgd():
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
    o.stochastic_gradient_descent(
        n, sin_data, learning_rate=1e-4, n_iters=10000, eval_every=1000
    )
    y_pred = n(x_pred)
    p.plot_1D_regression(
        "Results/1D sin predictions (trained)", sin_data, x_pred, y_pred,
    )

def train_2D4D_regression_sgd():
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
    o.stochastic_gradient_descent(
        n, sin_data, learning_rate=1e-4, n_iters=100, eval_every=10
    )
    y_pred = n(x_pred)
    p.plot_2D_nD_regression(
        "Results/2Dto4D sin predictions (trained)", 4, sin_data, y_pred
    )

def train_1D_regression_sgd_2way_tracking():
    # Test forward tracking
    sin_data = d.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)
    n = m.NeuralNetwork(
        # 1, 1, [10, 10],
        1, 1, [20],
        # act_funcs=[a.Relu(), a.Identity()]
    )
    x_pred = np.linspace(-3, 3, 200).reshape(1, -1)
    y_pred = n(x_pred)
    p.plot_1D_regression(
        "Results/1D sin predictions (untrained)", sin_data, x_pred, y_pred,
    )
    results = o.sgd_2way_tracking(
        n, sin_data, 10000, 1000
        # n_iters=20000, print_every=1000
    )
    y_pred = n(x_pred)
    p.plot_1D_regression(
        "Results/1D sin predictions (trained w ft)", sin_data, x_pred, y_pred,
    )
    p.plot_training_curves(
        [results], "SGD with line-search",
        e_lims=[0, 0.25], t_lims=None, i_lims=None
    )

def compare_sgd_sgd2w_learning_curves():
    sin_data = d.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)

    results_list = []
    for _ in range(3):
        n = m.NeuralNetwork(1, 1, [20])
        w = n.get_parameter_vector().copy()
        results_list.append(o.stochastic_gradient_descent(
            n, sin_data, 30000, 1500, learning_rate=1e-3,
            name="SGD, LR=1e-3"
        ))
        n.set_parameter_vector(w)
        results_list.append(o.stochastic_gradient_descent(
            n, sin_data, 30000, 1500, learning_rate=1e-1,
            name="SGD, LR=1e-1"
        ))
        n.set_parameter_vector(w)
        results_list.append(
            o.sgd_2way_tracking(n, sin_data, 10000, 500)
        )

    p.plot_training_curves(
        results_list, "SGD vs SCG-LS", e_lims=[0, 0.3])

def compare_logitistic_gaussian_afuncs(n_repeats=3, seed=0):
    np.random.seed(seed)
    sin_data = d.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)

    results_list = []
    for _ in range(n_repeats):
        # Initialise neural networks
        n_gaussian = m.NeuralNetwork(
            1, 1, [20], act_funcs=[a.Gaussian(), a.Identity()])
        n_logistic = m.NeuralNetwork(
            1, 1, [20], act_funcs=[a.Logistic(), a.Identity()])
        # Perform minimisation
        results_list.append(o.sgd_2way_tracking(
            n_gaussian, sin_data, 10000, 500,
            name="SGD-LS, Gaussian hidden activation"))
        results_list.append(o.sgd_2way_tracking(
            n_logistic, sin_data, 10000, 500,
            name="SGD-LS, logistic hidden activation"))
    
    p.plot_training_curves(
        results_list, "Logistic vs Gaussian activation functions",
        e_lims=[0, 0.3])

def plot_gaussian_sin_preds():
    # Create data
    sin_data = d.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)
    # Initialise network
    n_gaussian = m.NeuralNetwork(
        1, 1, [20], act_funcs=[a.Gaussian(), a.Identity()])
    # Perform minimisation
    o.sgd_2way_tracking(
        n_gaussian, sin_data, 10000, 500,
        name="SGD-LS, Gaussian hidden activation")
    # Create predictions
    x_pred = np.linspace(-3, 3, 200).reshape(1, -1)
    y_pred = n_gaussian(x_pred)
    # Plot results
    p.plot_1D_regression(
        "Results/1D sin predictions (Gaussian afunc)",
        sin_data, x_pred, y_pred)
    

if __name__ == "__main__":
    t_start = perf_counter()
    # Warm up numpy
    sin_data = d.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)
    n = m.NeuralNetwork(1, 1, [10])
    o.stochastic_gradient_descent(n, sin_data, 5000, 1000)
    # Set random seed
    np.random.seed(0)
    # Perform experiments:
    # train_1D_regression_sgd()
    # train_2D4D_regression_sgd()
    # train_1D_regression_sgd_2way_tracking()
    # compare_sgd_sgd2w_learning_curves()
    compare_logitistic_gaussian_afuncs()
    # plot_gaussian_sin_preds()

    # Print total running time
    t_total = perf_counter() - t_start
    print("---\nAll experiments completed in {:.2f} s".format(t_total))
