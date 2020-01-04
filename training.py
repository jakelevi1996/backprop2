import numpy as np
import matplotlib.pyplot as plt

import models as m, data as d, optimisers as o, plotting as p, \
    activations as a, errors as e

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
        n, sin_data, learning_rate=1e-4, n_iters=10000, print_every=1000
        # n_iters=20000, print_every=1000
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
        n, sin_data, learning_rate=1e-4, n_iters=100, print_every=10
        # n_iters=20000, print_every=1000
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
    train_errors, test_errors, times, iters, _ = o.sgd_2way_tracking(
        n, sin_data, n_iters=10000, print_every=1000, store_every=100
        # n_iters=20000, print_every=1000
    )
    y_pred = n(x_pred)
    p.plot_1D_regression(
        "Results/1D sin predictions (trained w ft)", sin_data, x_pred, y_pred,
    )
    p.plot_training_curves(
        "Results/SGDLS training curves",
        train_errors, test_errors, times, iters,
        e_lims=[0, 0.25], t_lims=None, i_lims=None
    )
    

if __name__ == "__main__":
    np.random.seed(0)
    # train_1D_regression_sgd()
    # train_2D4D_regression_sgd()
    train_1D_regression_sgd_2way_tracking()
