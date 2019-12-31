import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import data, models

def min_and_max(*input_arrays):
    """
    min_and_max: given a variable number of np.ndarrays, return the smallest and
    largest elements out of all of the input and output arguments
    """
    min_elem = min([array.min() for array in input_arrays])
    max_elem = max([array.max() for array in input_arrays])
    return min_elem, max_elem

def plot_1D_regression(
    filename, x_train, y_train, x_test, y_test, x_pred=None, y_pred=None,
    train_marker="bo", test_marker="ro", pred_marker="g-", tp=0.75,
    figsize=[8, 6], plot_title="1D regression data"
):
    """
    plot_1D_regression:
    TODO: write docstring
    """
    plt.figure(figsize=figsize)
    plt.plot(x_train, y_train, train_marker, alpha=tp)
    plt.plot(x_test, y_test, test_marker, alpha=tp)
    plot_preds = False if ((x_pred is None) or (y_pred is None)) else True
    if plot_preds:
        plt.plot(x_pred, y_pred, pred_marker, alpha=tp)
        plt.legend(["Training data", "Test data", "Predictions"])
    else:
        plt.legend(["Training data", "Test data"])
    plt.title(plot_title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_2D_nD_regression(
    filename, n_output_dims, dataset, y_pred, fig_title=None
):
    """
    plot_2D_nD_regression: 
    TODO: write docstring
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
            x_test0, x_test1,
            dataset.y_test[i].reshape(nx1, nx0),
            vmin=y_min, vmax=y_max
        )
        axes[1][i].pcolormesh(
            x_test0, x_test1,
            np.where(train_inds, dataset.y_test[i], np.nan).reshape(nx1, nx0),
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
    fig.suptitle(fig_title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename)
    plt.close()



def plot_2D_classification(self, filename, figsize=[8, 6]):
    pass
    # TODO: add plotting method for binary/discrete data

def plot_training_curves():
    pass

def plot_speed_test():
    pass

if __name__ == "__main__":
    np.random.seed(0)
    # Generate training and test sets for 1D to 1D regression
    s11 = data.SinusoidalDataSet11(n_train=100, n_test=50, xlim=[0, 1])
    # Generate untrained evaluations
    min_elem, max_elem = min_and_max(s11.x_train, s11.x_test)
    x_pred = np.linspace(min_elem, max_elem, 200).reshape(1, -1)
    n = models.NeuralNetwork(1, 1)
    y_pred = n(x_pred)
    # Plot
    plot_1D_regression(
        "Results/Untrained predictions 1D sinusoid",
        s11.x_train.ravel(), s11.y_train.ravel(), s11.x_test.ravel(),
        s11.y_test.ravel(), x_pred.ravel(), y_pred.ravel()
    )

    # Generate training and test sets for 2D to 3D regression
    s23 = data.SinusoidalDataSet2n(
        nx0=100, nx1=100, train_ratio=0.9, output_dim=4
    )
    # Generate untrained evaluations
    n = models.NeuralNetwork(2, 4)
    y_pred = n(s23.x_test)
    # Plot
    plot_2D_nD_regression(
        "Results/Untrained predictions 2D sinusoid", n_output_dims=4,
        dataset=s23, y_pred=y_pred
    )
    # print(s23.x_test.T)
