""" Module to contain all of the various plotting functions used in this
repository """
import os
from math import ceil, sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import PIL
import data, optimisers

def save_and_close(plot_name, dir_name, fig=None, file_ext="png"):
    """ Save and close the figure, first creating the output directory if it
    doesn't exist, and return the path to the output file """
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    file_name = "%s.%s" % (plot_name, file_ext)
    full_path = os.path.join(dir_name, file_name)
    if fig is None:
        plt.savefig(full_path)
        plt.close()
    else:
        fig.savefig(full_path)
        plt.close(fig)

    return full_path

def min_and_max(*input_arrays):
    """
    min_and_max: given a variable number of np.ndarrays, return the smallest and
    largest elements out of all of the input arguments
    """
    min_elem = min([array.min() for array in input_arrays])
    max_elem = max([array.max() for array in input_arrays])
    return min_elem, max_elem

def simple_plot(
    x,
    y,
    x_label,
    y_label,
    plot_name,
    dir_name,
    alpha=0.5,
    fmt="bo"
):
    """ Make a simple plot, with x and y data, axis labels, a title,
    configurable transparency and marker/line format, and save in an image file
    with the same name as the title, in the specified directory """
    # Create figure and plot
    plt.figure(figsize=[8, 6])
    plt.plot(x, y, fmt, alpha=alpha)
    # Format, save and close the figure
    plt.title(plot_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.grid(True)
    save_and_close(plot_name, dir_name)

def plot_1D_regression(
    plot_name,
    dir_name,
    dataset,
    x_pred=None,
    y_pred=None,
    train_marker="bo",
    test_marker="ro",
    pred_marker="g-",
    tp=0.75,
    figsize=[8, 6],
):
    """
    plot_1D_regression: plot the training data, test data, and optionally also
    model predictions, for a 1D regression data set. The dataset argument should
    be an instance of data.DataSet, and should contain x_train, y_train, x_test,
    and y_test attributes
    """
    assert dataset.input_dim == 1
    plt.figure(figsize=figsize)
    # Plot training and test data
    plt.plot(
        dataset.x_train.ravel(),
        dataset.y_train.ravel(),
        train_marker,
        alpha=tp
    )
    plt.plot(
        dataset.x_test.ravel(),
        dataset.y_test.ravel(),
        test_marker,
        alpha=tp
    )
    if (x_pred is not None) and (y_pred is not None):
        # Plot predictions
        plt.plot(x_pred.ravel(), y_pred.ravel(), pred_marker, alpha=tp)
        plt.legend(["Training data", "Test data", "Predictions"])
    else:
        plt.legend(["Training data", "Test data"])
    # Format, save and close
    plt.title(plot_name)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    save_and_close(plot_name, dir_name)

def plot_2D_nD_regression(
    plot_name,
    dir_name,
    n_output_dims,
    dataset,
    x_pred_0,
    x_pred_1,
    model,
    tight_layout=False
):
    """
    Plot the training data, test data, and model predictions for a regression
    data set with 2 input dimensions and n_output_dims output dimensions.

    Inputs:
    -   plot_name: title of the plot; will also be used as the filename
    -   dir_name: name of directory to save plot to (will be created if it
        doesn't already exist)
    -   n_output_dims: number of output dimensions to plot
    -   dataset: should be an instance of data.DataSet, and should contain
        x_train, y_train, x_test, and y_test attributes
    -   x_pred_0: first dimension of inputs that the model will use to make
        predictions, as a 1-dimensional np.ndarray. The upper and lower limits
        of this array will be used to set the x axis limits
    -   x_pred_1: second dimension of inputs that the model will use to make
        predictions, as a 1-dimensional np.ndarray. The upper and lower limits
        of this array will be used to set the y axis limits
    -   model: instance of NeuralNetwork, used to form predictions
    -   tight_layout: Boolean flag indicating whether or not to use a
        tight-layout for saving the plots, which (for some reason) makes this
        function about 50% slower for certain inputs. Default is False.
    """
    assert dataset.input_dim == 2
    # Create subplots and set figure size
    fig, axes = plt.subplots(
        3,
        n_output_dims + 1,
        sharex=True,
        sharey=True,
        gridspec_kw={"width_ratios": ([1] * n_output_dims) + [0.2]}
    )
    fig.set_size_inches(4 * (n_output_dims + 1), 10)
    if axes.ndim == 1:
        axes = np.expand_dims(axes, 1)
    y_min = dataset.y_test.min()
    y_max = dataset.y_test.max()
    # Use model to make predictions
    xx0, xx1 = np.meshgrid(x_pred_0, x_pred_1)
    x_pred = np.stack([xx0.ravel(), xx1.ravel()], axis=0)
    y_pred = model(x_pred)
    # Iterate through each output dimension
    for i in range(n_output_dims):
        # Plot training data
        axes[0][i].scatter(
            dataset.x_train[0, :],
            dataset.x_train[1, :],
            c=dataset.y_train[i, :],
            vmin=y_min,
            vmax=y_max,
            alpha=0.5,
            ec=None
        )
        # Plot test data
        axes[1][i].scatter(
            dataset.x_test[0, :],
            dataset.x_test[1, :],
            c=dataset.y_test[i, :],
            vmin=y_min,
            vmax=y_max,
            alpha=0.5,
            ec=None
        )
        # Plot predictions
        axes[2][i].pcolormesh(
            x_pred_0,
            x_pred_1,
            y_pred[i].reshape(x_pred_1.size, x_pred_0.size),
            vmin=y_min,
            vmax=y_max
        )
        axes[2][i].set_xlabel("$y_{}$".format(i))
    # Format, save and close
    axes[0][0].set_ylabel("Training data")
    axes[1][0].set_ylabel("Test data")
    axes[2][0].set_ylabel("Predictions")
    for i in range(axes.shape[0]):
        axes[i, -1].set_ylim(x_pred_1.min(), x_pred_1.max())
    for j in range(axes.shape[1]):
        axes[-1, j].set_xlim(x_pred_0.min(), x_pred_0.max())
    for a in axes[:, -1]:
        a.axis("off")
    fig.colorbar(
        ScalarMappable(Normalize(y_min, y_max)),
        ax=axes[:, -1],
        fraction=1
    )
    fig.suptitle(plot_name, fontsize=16)
    if tight_layout:
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_and_close(plot_name, dir_name, fig)

def plot_1D_layer_acts(filename, neural_network, xlims=[-1, 1]):
    """
    Function to plot the activations of each hidden and output unit in a neural
    network, with a 1-dimensional input
    """
    raise NotImplementedError

def plot_2D_classification(self, filename, figsize=[8, 6]):
    pass
    # TODO: add plotting method for binary/discrete data

def plot_training_curves(
    result_list,
    plot_name="Learning curves",
    dir_name="Results/Learning curves",
    figsize=[15, 6],
    e_lims=None,
    t_lims=None,
    i_lims=None,
    tp=0.75,
    error_log_axis=False,
    time_log_axis=False,
    iter_log_axis=False
):
    """
    Given a list of Result objects, create a plot containing the training and
    test errors for each Result, both against time and against number of
    iterations, and also plot the number of iterations against time. The plot is
    saved in the specified file and directory. Multiple repeats of Result
    objects with the same name attribute are supported, and plotted in the same
    colour, and with a single legend entry, for easy comparison.

    TODO:
    -   2*2 subplots with legend in its own subplot?
    -   Logarithmic axes
    """
    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(figsize)
    name_list = [result.name for result in result_list]
    unique_names_list = sorted(list(set(name_list)))
    colour_list = plt.get_cmap("hsv")(
        np.linspace(0, 1, len(unique_names_list), endpoint=False)
    )
    colour_dict = dict(zip(unique_names_list, colour_list))
    for result in result_list:
        # Get line colour, depending on the name of the experiment
        colour = colour_dict[result.name]
        # Get values from Result object
        times = result.get_values(optimisers.results.columns.Time)
        train_errors = result.get_values(optimisers.results.columns.TrainError)
        test_errors = result.get_values(optimisers.results.columns.TestError)
        iters = result.get_values(optimisers.results.columns.Iteration)
        # Plot errors against time
        axes[0].plot(times, train_errors,   c=colour, ls="--",  alpha=tp)
        axes[0].plot(times, test_errors,    c=colour, ls="-",   alpha=tp)
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Mean error")
        axes[0].grid(True)
        # Plot errors against iteration
        axes[1].plot(iters, train_errors,   c=colour, ls="--",  alpha=tp)
        axes[1].plot(iters, test_errors,    c=colour, ls="-",   alpha=tp)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Mean error")
        axes[1].grid(True)
        # Plot iteration against time
        axes[2].plot(times, iters, c=colour, ls="-", alpha=tp)
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Iteration")
        axes[2].grid(True)
    
    # Set axis limits
    if e_lims is not None:
        axes[0].set_ylim(*e_lims)
        axes[1].set_ylim(*e_lims)
    if t_lims is not None:
        axes[0].set_xlim(*t_lims)
        axes[2].set_xlim(*t_lims)
    if i_lims is not None:
        axes[1].set_xlim(*i_lims)
        axes[2].set_ylim(*i_lims)

    # Format, save and close
    handles = []
    for colour, result_name in zip(colour_list, unique_names_list):
        handles.append(Line2D([], [], c=colour, ls="-", alpha=tp,
            label=result_name))
    handles += [
        Line2D([], [], c="k", ls="-", alpha=tp, label="Test error"),
        Line2D([], [], c="k", ls="--", alpha=tp, label="Train error")]
    axes[2].legend(handles=handles)
    fig.suptitle(plot_name)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_and_close(plot_name.replace("\n", ", "), dir_name, fig)

def plot_speed_trials():
    pass

def plot_act_func(act_func, dir_name, xlims, npoints):
    """
    Plot an activation function and its derivatives
    """
    x = np.linspace(*xlims, npoints)
    y = act_func.y(x)
    dydx = act_func.dydx(x)
    d2ydx2 = act_func.d2ydx2(x)
    plt.figure(figsize=[8, 6])
    plt.plot(x, y, 'b', x, dydx, 'r', x, d2ydx2, 'g', alpha=0.75)
    plt.legend(["$y(x)$", "$\\frac{dy}{dx}(x)$", "$\\frac{d^2y}{dx^2}(x)$"])
    plt.title(act_func.name)
    plt.grid(True)
    save_and_close(act_func.name, dir_name)

def plot_error_func(error_func, dir_name, xlims, npoints):
    """
    Plot an error function and its derivatives
    """
    y = np.linspace(*xlims, npoints).reshape(1, -1)
    t = 0
    E = error_func.E(y, 0)
    dEdy = error_func.dEdy(y, 0)
    d2Edy2 = error_func.d2Edy2(y, 0)
    plt.figure(figsize=[8, 6])
    plt.plot(y.ravel(), E.ravel(), 'b', alpha=0.75)
    plt.plot(y.ravel(), dEdy.ravel(), 'r', alpha=0.75)
    plt.plot(y.ravel(), d2Edy2.ravel(), 'g', alpha=0.75)
    plt.axvline(0, c="k", ls="--", alpha=0.75)
    plt.legend([
        "$E(y, t)$",
        "$\\frac{dE}{dy}(y, t)$",
        "$\\frac{d^2E}{dy^2}(y, t)$",
        "Target $t = 0.0$"
    ])
    plt.title(error_func.name)
    plt.grid(True)
    save_and_close(error_func.name, dir_name)

def plot_result_attribute(
    plot_name,
    dir_name,
    result_list,
    attribute,
    figsize=[8, 6],
    alpha=0.7,
    marker=None,
    line_style=None
):
    """ Function to plot a specific attribute stored in the Result class, for
    example the step size, or the DBS, during each iteration. attribute should
    be the type of the attribute which will be retrieved from the Result object,
    and should be a subclass of optimisers.results.columns._Column, EG
    optimisers.results.columns.StepSize """
    plt.figure(figsize=figsize)
    name_list = [result.name for result in result_list]
    unique_names_list = sorted(list(set(name_list)))
    colour_list = plt.get_cmap("hsv")(
        np.linspace(0, 1, len(unique_names_list), endpoint=False)
    )
    colour_dict = dict(zip(unique_names_list, colour_list))
    for result in result_list:
        plt.plot(
            result.get_values(optimisers.results.columns.Iteration),
            result.get_values(attribute),
            c=colour_dict[result.name],
            alpha=alpha,
            marker=marker,
            ls=line_style
        )
    # Format, save and close
    plt.title(plot_name)
    plt.xlabel("Iteration")
    plt.ylabel(result_list[0].get_column_name(attribute))
    plt.legend(handles=[
        Line2D([], [], color=c, label=name, marker=marker, ls=line_style)
        for c, name in zip(colour_list, unique_names_list)
    ])
    plt.grid(True)
    save_and_close(plot_name.replace("\n", ", "), dir_name)

def plot_result_attributes_subplots(
    plot_name,
    dir_name,
    result_list,
    attribute_list,
    num_rows=None,
    num_cols=None,
    figsize=[16, 9],
    alpha=0.7,
    marker=None,
    line_style=None,
    log_axes_attributes=None
):
    """ Similar to the plot_result_attribute function, except accept a list of
    attribute types, and use a different subplot for each attribute (and one
    also for the legend). If num_rows or num_cols are not set, then they are
    chosen to make the plot as square as possible. If present,
    log_axes_attributes should be an iterable (EG a set) containing the types of
    any attributes for which the corresponding subplot should have logarithmic y
    axes.

    Raises a ValueError if num_rows * num_cols < len(attribute_list) + 1.
    """
    # Set number of plots, rows and columns, and log_axes_attributes
    num_plots = len(attribute_list) + 1
    if num_rows is None and num_cols is None:
        num_cols = ceil(sqrt(num_plots))
    if num_rows is None:
        num_rows = ceil(num_plots / num_cols)
    if num_cols is None:
        num_cols = ceil(num_plots / num_rows)
    if num_rows * num_cols < num_plots:
        raise ValueError("Not enough rows/columns for attribute list")
    if log_axes_attributes is None:
        log_axes_attributes = []

    # Create subplots, name list, colour list, and colour dictionary
    fig, axes = plt.subplots(num_rows, num_cols, sharex=True, figsize=figsize)
    name_list = [result.name for result in result_list]
    unique_names_list = sorted(list(set(name_list)))
    colour_list = plt.get_cmap("hsv")(
        np.linspace(0, 1, len(unique_names_list), endpoint=False)
    )
    colour_dict = dict(zip(unique_names_list, colour_list))
    # Iterate through attributes, axes, and results
    for attribute, ax in zip(attribute_list, axes.flat):
        if attribute in log_axes_attributes:
            ax_plot_func = lambda *args, **kwargs: ax.semilogy(*args, **kwargs)
        else:
            ax_plot_func = lambda *args, **kwargs: ax.plot(*args, **kwargs)
        for result in result_list:
            ax_plot_func(
                result.get_values(optimisers.results.columns.Iteration),
                result.get_values(attribute),
                c=colour_dict[result.name],
                alpha=alpha,
                marker=marker,
                ls=line_style
            )
        ax.set_xlabel("Iteration")
        ax.set_ylabel(result_list[0].get_column_name(attribute))
        ax.grid(which="major", ls="-")
        ax.grid(which="minor", ls=":", alpha=0.5)

    # Format, save and close
    fig.suptitle(plot_name, fontsize=20)
    legend_subplot_index = len(attribute_list)
    axes.flat[legend_subplot_index].legend(
        loc="center",
        handles=[
            Line2D([], [], color=c, label=name, marker=marker, ls=line_style)
            for c, name in zip(colour_list, unique_names_list)
        ]
    )
    axes.flat[legend_subplot_index].axis("off")
    save_and_close(plot_name.replace("\n", ", "), dir_name, fig)

def plot_error_reductions_vs_batch_size(
    plot_name,
    dir_name,
    reduction_dict,
    figsize=[16, 6],
    y_lim_left=None,
    y_lim_right=None,
    n_sigma=2
):
    """ Function to plot statistics for the reduction in the mean error in the
    test set after a single minimisation iteration as a function of the batch
    size used for the iteration. This information is represented in two
    subplots, one for the reduction in the test set error, and one for the ratio
    of the reduction in the test set error to the batch size used for the
    iteration, which indicates the "efficiency" of the iteration. The motivation
    for plotting this information is that typically over the course of
    minimisation, we want to reduce the mean test set error as much as possible
    as fast as possible; using a large batch size will give more reliably large
    reductions, however will also take longer for each iteration.

    Inputs:
    -   plot_name: name of the plot, also used as the filename
    -   dir_name: name of the directory to save the plot in
    -   reduction_dict: dictionary in which each key is an integer batch-size,
        and each value is a corresponding list of test set reductions
    -   fig_size: size of the figure in inches
    -   y_lim_left: limits for y axes for the left subplot
    -   y_lim_right: limits for y axes for the right subplot
    -   n_sigma: number of standard deviations away from the mean to plot.
        Default is 2

    Example usage: see function test_plot_error_reductions_vs_batch_size in
    Tests/test_plotting.py

    TODO: logarithmic x-axis (batch size)?
    """
    # Initialise the figure, axes, and format dictionaries
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=figsize)
    std_fmt = {
        "color": "b",
        "alpha": 0.3,
        "zorder": 10,
        "label": "$\\pm%i\\sigma$" % n_sigma
    }
    data_point_fmt = {
        "color": "k",
        "marker": "o",
        "ls": "",
        "label": "Single data point",
        "alpha": 0.5,
        "zorder": 20
    }
    mean_fmt = {"c": "b", "ls": "--", "label": "Mean reduction", "zorder": 30}
    # Calculate mean and standard deviation
    batch_size_list = np.array(sorted(reduction_dict.keys()))
    mean = np.array([
        np.mean(reduction_dict[batch_size])
        for batch_size in batch_size_list
    ])
    std = np.array([
        np.std(reduction_dict[batch_size])
        for batch_size in batch_size_list
    ])
    # Plot the standard deviations
    axes[0].fill_between(
        batch_size_list,
        mean + n_sigma*std,
        mean - n_sigma*std,
        **std_fmt
    )
    axes[1].fill_between(
        batch_size_list,
        (mean + n_sigma*std) / batch_size_list,
        (mean - n_sigma*std) / batch_size_list,
        **std_fmt
    )
    # Plot each individual data point using a semi-transparent marker
    b_list_repeated, r_list_unpacked = zip(*[
        [b, r]
        for b, r_list in reduction_dict.items()
        for r in r_list
    ])
    r_over_b_list = np.array(r_list_unpacked) / np.array(b_list_repeated)
    axes[0].plot(b_list_repeated, r_list_unpacked, **data_point_fmt)
    axes[1].plot(b_list_repeated, r_over_b_list, **data_point_fmt)

    # Plot the means
    axes[0].plot(batch_size_list, mean, **mean_fmt)
    axes[1].plot(batch_size_list, mean / batch_size_list, **mean_fmt)

    # Format, save and close
    fig.suptitle(plot_name, fontsize=15)
    no_reduction_fmt = {"color": "r", "ls": "-", "label": "No reduction"}
    for a in axes:
        a.axhline(0, **no_reduction_fmt)
        a.grid(True)
        a.set_xlim(0, max(batch_size_list))
        a.set_xlabel("Batch size")
    axes[0].set_ylim(y_lim_left)
    axes[1].set_ylim(y_lim_right)
    axes[0].set_ylabel("Mean test set error reduction")
    axes[1].set_ylabel(
        "$\\frac{\\mathrm{Mean\\/test\\/set\\/error\\/reduction}}"
        "{\\mathrm{Batch\\/size}}$"
    )
    axes[1].legend(
        handles=[
            Line2D([], [], **data_point_fmt),
            Line2D([], [], **mean_fmt),
            Patch(**std_fmt),
            Line2D([], [], **no_reduction_fmt),
        ],
        loc="upper right"
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    full_path = save_and_close(plot_name.replace("\n", ", "), dir_name, fig)
    return full_path


def make_gif(
    output_name,
    output_dir,
    input_path_list,
    duration=100,
    optimise=False,
    loop=0
):
    """ Make gif using pre-existing image files, and save to disk. The gif will
    loop indefinitely.

    Usage example: see Tests/test_plotting.py

    Inputs:
    -   output_name: filename for the output gif (not including .gif file
        extension)
    -   output_dir: directory that the output gif will be saved to
    -   input_path_list: list of file names of images, each of which will form a
        single frame of the gif, in the order that they're specifed in the list.
        The file names should include the file extension (EG .png), as well as
        the directory name (if not in the current directory)
    -   duration: the duration each frame of the gif should last for, in
        milliseconds. Default is 100 seconds
    -   optimise: if True, attempt to compress the palette by eliminating unused
        colors. Default is False
    -   loop: integer number of times the GIF should loop. 0 means that it will
        loop forever. If None, then the image will not loop at all. By default,
        the image will loop forever
    """
    first_frame = PIL.Image.open(input_path_list[0])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_filename = "%s.gif" % output_name
    output_path = os.path.join(output_dir, output_filename)
    loop_kwarg = dict() if (loop is None) else {"loop": loop}
    first_frame.save(
        output_path,
        format="gif",
        save_all=True,
        append_images=[PIL.Image.open(f) for f in input_path_list[1:]],
        duration=duration,
        optimise=optimise,
        **loop_kwarg
    )

def plot_parameter_sweep_results(
    experiment_results,
    param_name,
    plot_name,
    dir_name,
    n_sigma=2
):
    """ Plot the results of running experiments to sweep over various different
    values of a parameter, with multuple repeats of each experiment.

    Inputs:
    -   experiment_results: a dictionary in which each dictionary-key is a value
        for the parameter-under-test, and each dictionary-value is a list of the
        final test set errors for each repeat of the experiment with
        parameter-under-test taking the corresponding value
    -   param_name: name of the parameter under test (used as the x-axis label)
    -   plot_name: title for the plot, and filename that the output image is
        saved to
    -   dir_name: directory in which to save the output image
    -   n_sigma: number of standard deviations to plot (as a semi-transparent
        polygon) above and below the mean
    """
    # Create figure and format dictionaries
    plt.figure(figsize=[8, 6])
    std_fmt = {
        "color": "b",
        "alpha": 0.3,
        "zorder": 10,
        "label": "$\\pm%i\\sigma$" % n_sigma
    }
    data_point_fmt = {
        "color": "k",
        "marker": "o",
        "ls": "",
        "label": "Result",
        "alpha": 0.5,
        "zorder": 20
    }
    mean_fmt = {"c": "b", "ls": "--", "label": "Mean", "zorder": 30}
    
    # Calculate mean and standard deviation, in ordered numpy arrays
    val_list = sorted(list(experiment_results.keys()))
    mean = np.array([np.mean(experiment_results[val]) for val in val_list])
    std  = np.array([np.std( experiment_results[val]) for val in val_list])

    # Plot data
    for val, error_list in experiment_results.items():
        for error in error_list:
            plt.plot(val, error, **data_point_fmt)
    plt.plot(val_list, mean, **mean_fmt)
    plt.fill_between(
        val_list,
        mean + n_sigma*std,
        mean - n_sigma*std,
        **std_fmt
    )
    
    # Format, save and close the figure
    plt.title(plot_name)
    plt.xlabel(param_name)
    plt.ylabel("Final test set error")
    plt.legend(
        handles=[
            Line2D([], [], **data_point_fmt),
            Line2D([], [], **mean_fmt),
            Patch(**std_fmt),
        ],
    )
    plt.tight_layout()
    plt.grid(True)
    save_and_close(plot_name, dir_name)
