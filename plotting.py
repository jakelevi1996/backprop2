import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import data

def min_and_max(*input_arrays):
    """
    min_and_max: given a variable number of np.ndarrays, return the smallest and
    largest elements out of all of the input and output arguments
    """
    min_elem = min([array.min() for array in input_arrays])
    max_elem = max([array.max() for array in input_arrays])
    return min_elem, max_elem

def plot_1D_regression(
    filename, dataset, x_pred=None, y_pred=None, train_marker="bo",
    test_marker="ro", pred_marker="g-", tp=0.75, figsize=[8, 6],
    fig_title="1D regression data", fig_title_append=None
):
    """
    plot_1D_regression: plot the training data, test data, and optionally also
    model predictions, for a 1D regression data set. The dataset argument should
    be an instance of data.DataSet, and should contain x_train, y_train, x_test,
    and y_test attributes
    """
    plt.figure(figsize=figsize)
    # Plot training and test data
    plt.plot(
        dataset.x_train.ravel(), dataset.y_train.ravel(), train_marker, alpha=tp
    )
    plt.plot(
        dataset.x_test.ravel(), dataset.y_test.ravel(), test_marker, alpha=tp
    )
    plot_preds = False if ((x_pred is None) or (y_pred is None)) else True
    if plot_preds:
        # Plot predictions
        plt.plot(x_pred.ravel(), y_pred.ravel(), pred_marker, alpha=tp)
        plt.legend(["Training data", "Test data", "Predictions"])
    else:
        plt.legend(["Training data", "Test data"])
    if fig_title_append is not None:
        fig_title += fig_title_append
    # Format, save and close
    plt.title(fig_title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_2D_nD_regression(
    filename, n_output_dims, dataset, y_pred, fig_title=None,
    fig_title_append=None
):
    """
    plot_2D_nD_regression: plot the training data, test data, and model
    predictions for a regression data set with 2 input dimensions and
    n_output_dims output dimensions.

    Inputs:
    -   filename: string containing the filename that the plot should be saved
        to
    -   n_output_dims: number of output dimensions to plot
    -   dataset: should be an instance of data.DataSet, and should contain
        x_train, y_train, x_test, and y_test attributes. It is assumed that
        x_test is a uniform 2D grid created using np.meshgrid, and x_train is a
        subset of the points contained in x_test
    -   y_pred: 
    """
    # Create subplots and set figure size
    fig, axes = plt.subplots(3, n_output_dims, sharex=True, sharey=True)
    fig.set_size_inches(4 * n_output_dims, 10)
    # Reshape test and evaluation data back into grids
    nx0 = np.unique(dataset.x_test[0]).size
    nx1 = np.unique(dataset.x_test[1]).size
    x_test0 = dataset.x_test[0].reshape(nx1, nx0)
    x_test1 = dataset.x_test[1].reshape(nx1, nx0)
    y_min, y_max = dataset.y_test.min(), dataset.y_test.max()
    train_inds = np.equal(
        dataset.x_test.T.reshape(1, -1, 2),
        dataset.x_train.T.reshape(-1, 1, 2)
    ).all(axis=2).any(axis=0)
    # Plot test set, training set, and evaluations
    for i in range(n_output_dims):
        axes[0][i].pcolormesh(
            x_test0, x_test1, dataset.y_test[i].reshape(nx1, nx0),
            vmin=y_min, vmax=y_max
        )
        y_train = np.where(train_inds, dataset.y_test[i], np.nan)
        axes[1][i].pcolormesh(
            x_test0, x_test1, np.reshape(y_train, [nx1, nx0]),
            vmin=y_min, vmax=y_max
        )
        axes[2][i].pcolormesh(
            x_test0, x_test1, y_pred[i].reshape(nx1, nx0),
            vmin=y_min, vmax=y_max
        )
        axes[2][i].set_xlabel("y[{}]".format(i))
    # Format, save and close
    cm.get_cmap().set_bad("k")
    axes[0][0].set_ylabel("Test data")
    axes[1][0].set_ylabel("Training data")
    axes[2][0].set_ylabel("Predictions")
    if fig_title is None:
        fig_title = "2D to {}D regression data".format(n_output_dims)
    if fig_title_append is not None:
        fig_title += fig_title_append
    fig.suptitle(fig_title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename)
    plt.close()

def plot_1D_layer_acts(filename, neural_network, xlims=[-1, 1]):
    raise NotImplementedError

def plot_2D_classification(self, filename, figsize=[8, 6]):
    pass
    # TODO: add plotting method for binary/discrete data

def plot_training_curves(
    filename, train_errors, test_errors, times, iters, figsize=[15, 6],
    e_lims=[0, 0.5], t_lims=None, i_lims=None, tp=0.75
):
    # TODO: update this to match outputs from optimiser functions, and comment,
    # and accept multiple different training curves (EG from different
    # optimisers) and legends (in a seperate subplot?)
    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(figsize)
    # Plot errors against time
    axes[0].plot(times, train_errors, 'b--', times, test_errors, 'b-', alpha=tp)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Mean error")
    axes[0].grid(True)
    if t_lims is not None: axes[0].set_xlim(*t_lims)
    if e_lims is not None: axes[0].set_ylim(*e_lims)
    # Plot errors against iteration
    axes[1].plot(iters, train_errors, 'b--', iters, test_errors, 'b-', alpha=tp)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Mean error")
    axes[1].grid(True)
    if i_lims is not None: axes[1].set_xlim(*i_lims)
    if e_lims is not None: axes[1].set_ylim(*e_lims)
    # Plot iteration against time
    axes[2].plot(times, iters, 'b--', times, iters, 'b-', alpha=tp)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Iteration")
    axes[2].grid(True)
    if t_lims is not None: axes[2].set_xlim(*t_lims)
    if i_lims is not None: axes[2].set_ylim(*i_lims)
    # Format, save and close
    fig.suptitle("Learning curves")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename)
    plt.close()

def plot_speed_trials():
    pass

def plot_act_func(filename, act_func, xlims, npoints):
    """
    plot_act_func: plot an activation function and its derivatives
    """
    x = np.linspace(*xlims, npoints)
    y = act_func.y(x)
    dydx = act_func.dydx(x)
    plt.figure(figsize=[8, 6])
    plt.plot(x, y, 'b', x, dydx, 'r', alpha=0.75)
    plt.legend(["Function", "Derivative"])
    plt.title(act_func.name)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    np.random.seed(0)
    # Generate training and test sets for 1D to 1D regression
    s11 = data.SinusoidalDataSet1D1D(n_train=100, n_test=50, xlim=[0, 1])
    # Generate random predictions
    min_elem, max_elem = min_and_max(s11.x_train, s11.x_test)
    N_D = 200
    x_pred = np.linspace(min_elem, max_elem, N_D).reshape(1, -1)
    y_pred = np.random.normal(size=[1, N_D])
    # Plot
    plot_1D_regression(
        "Results/Random predictions 1D sinusoid", s11,
        x_pred.ravel(), y_pred.ravel(), pred_marker="go"
    )

    # Generate training and test sets for 2D to 3D regression
    output_dim = 4
    s23 = data.SinusoidalDataSet2DnD(
        nx0=100, nx1=100, train_ratio=0.9, output_dim=output_dim
    )
    # Generate random predictions
    y_pred = np.random.normal(size=[output_dim, s23.x_test.shape[1]])
    # Plot
    plot_2D_nD_regression(
        "Results/Random predictions 2D sinusoid", n_output_dims=4,
        dataset=s23, y_pred=y_pred
    )
    # print(s23.x_test.T)
