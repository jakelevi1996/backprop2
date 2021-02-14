"""
Module containing unit tests for the plotting module. The functions for plotting
activation and error functions are not tested here, but in the test modules for
the activation and error function modules. Similarly, the function for plotting
training curves is tested in the test module for the optimiser module.
"""
import os
import numpy as np
from math import ceil
import pytest
import plotting, data, optimisers, models
from .util import get_random_network, get_output_dir

# Get name of output directory, and create it if it doesn't exist
output_dir = get_output_dir("Plotting")

@pytest.mark.parametrize("seed", [8974, 4798, 1812])
def test_plot_1D_regression(seed):
    """
    Test plotting function for data with 1 input dimension and 1 output
    dimension
    """
    np.random.seed(seed)
    # Generate training and test sets for 1D to 1D regression
    s11 = data.Sinusoidal(n_train=100, n_test=50, x_lo=0, x_hi=1)
    # Generate random predictions
    min_elem, max_elem = plotting.min_and_max(s11.x_train, s11.x_test)
    N_D = 200
    x_pred = np.linspace(min_elem, max_elem, N_D).reshape(1, -1)
    y_pred = np.random.normal(size=[1, N_D])
    # Call plotting function under test
    plotting.plot_1D_regression(
        "Random predictions 1D sinusoid",
        output_dir,
        s11,
        x_pred.ravel(),
        y_pred.ravel(),
        pred_marker="go"
    )


@pytest.mark.parametrize("seed, output_dim", [(1814, 1), (1743, 3)])
def test_plot_2D_nD_regression(seed, output_dim):
    """
    Test plotting function for data with 2 input dimensions and a variable
    number of output dimensions
    """
    np.random.seed(seed)
    input_dim = 2
    x_lo = -2
    x_hi = 2
    x_pred = np.linspace(x_lo, x_hi, 10)
    # Generate dataset and network
    sin_data = data.Sinusoidal(
        input_dim=input_dim,
        output_dim=output_dim,
        n_train=200,
        x_lo=x_lo,
        x_hi=x_hi
    )
    model = get_random_network(
        input_dim=input_dim,
        output_dim=output_dim,
        low=2,
        high=3,
        initialiser=models.initialisers.ConstantPreActivationStatistics(
            x_train=sin_data.x_train,
            y_train=sin_data.y_train
        )
    )
    # Call plotting function under test
    plotting.plot_2D_nD_regression(
        "Random predictions 2D-{}D sinusoid".format(output_dim),
        output_dir,
        n_output_dims=output_dim,
        dataset=sin_data,
        x_pred_0=x_pred,
        x_pred_1=x_pred,
        model=model,
    )

def test_plot_training_curves():
    np.random.seed(79)
    n_models = np.random.randint(2, 5)
    results_list = []

    for j in range(n_models):
        n_iters = np.random.randint(10, 20)
        output_dim = np.random.randint(2, 5)
        n = get_random_network(input_dim=2, output_dim=output_dim)
        d = data.Sinusoidal(input_dim=2, output_dim=output_dim, n_train=150)
        w = n.get_parameter_vector()
        result = optimisers.Result(name="Network {}".format(j))
        result.begin()
        
        # Call the result.update method a few times
        for i in range(n_iters):
            n.set_parameter_vector(w + i)
            result.update(model=n, dataset=d, iteration=i)
        
        results_list.append(result)
    
    plotting.plot_training_curves(
        results_list,
        "Test plot_training_curves",
        output_dir,
        e_lims=None
    )

def test_plot_result_attribute():
    """
    Test plotting function for plotting the values in one of the columns of a
    Result object over time
    """
    np.random.seed(1521)
    n_its = np.random.randint(10, 20)
    results_list = []
    for i in range(5):
        i = min(i, 2)
        name = "test_plot_result_attribute_%i" % i
        output_text_filename = os.path.join(output_dir, name + ".txt")
        with open(output_text_filename, "w") as f:
            result = optimisers.Result(
                name=name,
                file=f,
                add_default_columns=False
            )
            ls = optimisers.LineSearch()
            ls_column = optimisers.results.columns.StepSize(ls)
            result.add_column(ls_column)
            result.add_column(optimisers.results.columns.Iteration())
            result.begin()
            for j in range(n_its):
                ls.s = np.random.uniform() + i
                result.update(iteration=j)
        
        results_list.append(result)
    
    plotting.plot_result_attribute(
        "test_plot_result_attribute_linesearch",
        output_dir,
        results_list,
        attribute=type(ls_column),
        marker="o",
        line_style=""
    )

def test_plot_result_attribute_subplots():
    """
    Test plotting function for plotting the values in multiple columns of a
    Result object over time, with one subplot per column
    """
    np.random.seed(1521)
    n_its = np.random.randint(10, 20)
    n_train = np.random.randint(10, 20)
    sin_data = data.Sinusoidal(input_dim=1, output_dim=1, n_train=n_train)
    results_list = []
    for i in range(5):
        model = models.NeuralNetwork(input_dim=1, output_dim=1)
        model.get_gradient_vector(sin_data.x_train, sin_data.y_train)
        name = "test_plot_result_attribute_subplots_%i" % (i + 1)
        output_text_filename = os.path.join(output_dir, name + ".txt")
        with open(output_text_filename, "w") as f:
            result = optimisers.Result(name=name, file=f)
            ls = optimisers.LineSearch()
            ls_column = optimisers.results.columns.StepSize(ls)
            dbs_metric_column = optimisers.results.columns.DbsMetric()
            result.add_column(ls_column)
            result.add_column(dbs_metric_column)
            optimisers.gradient_descent(
                model,
                sin_data,
                result=result,
                line_search=ls,
                terminator=optimisers.Terminator(i_lim=n_its),
                evaluator=optimisers.Evaluator(i_interval=1)
            )
        results_list.append(result)
    
    attribute_list = [
        optimisers.results.columns.TrainError,
        optimisers.results.columns.TestError,
        optimisers.results.columns.Time,
        type(ls_column),
        type(dbs_metric_column)
    ]
    plotting.plot_result_attributes_subplots(
        "test_plot_result_attribute_subplots",
        output_dir,
        results_list,
        attribute_list,
        marker="o",
        line_style="--",
        log_axes_attributes={
            optimisers.results.columns.TrainError,
            optimisers.results.columns.TestError,
            type(ls_column)
        }
    )


def test_plot_error_reductions_vs_batch_size():
    """ Test function which plots statistics for the reduction in the mean error
    in the test set after a single minimisation iteration as a function of the
    batch size used for the iteration """
    np.random.seed(1753)
    # Initialise parameters for the test
    min_batch_size = 5
    n_batch_sizes = 10
    input_dim = 1
    output_dim = 1
    n_train = 100
    n_hidden_units = [10]
    title = "Test plotting error reductions vs batch size"

    # Initialise objects and lists
    model = models.NeuralNetwork(input_dim, output_dim, n_hidden_units)
    sin_data = data.Sinusoidal(input_dim, output_dim, n_train)
    w_0 = model.get_parameter_vector().copy()
    model.forward_prop(sin_data.x_test)
    E_0 = model.mean_error(sin_data.y_test)
    result = optimisers.Result(verbose=False, add_default_columns=False)
    evaluator = optimisers.DoNotEvaluate()
    terminator = optimisers.Terminator(i_lim=1)
    # TODO: logarithmically spaced batch sizes?
    batch_size_list = np.linspace(min_batch_size, n_train, n_batch_sizes)
    reduction_dict = dict()
    # Iterate through batch sizes
    for batch_size in batch_size_list:
        # Set number of repeats and initialise results list
        n_repeats = 10
        reduction_dict[batch_size] = []
        batch_getter = optimisers.batch.ConstantBatchSize(int(batch_size))
        # Iterate through repeats
        for _ in range(n_repeats):
            # Reset parameters and perform one iteration of gradients descent
            model.set_parameter_vector(w_0)
            result_ls = optimisers.gradient_descent(
                model,
                sin_data,
                batch_getter=batch_getter,
                terminator=terminator,
                evaluator=evaluator,
                result=result,
                display_summary=False
            )
            # Calculate new error and add the reduction to the list
            model.forward_prop(sin_data.x_test)
            error_reduction = E_0 - model.mean_error(sin_data.y_test)
            reduction_dict[batch_size].append(error_reduction)

    # Call function being tested
    plotting.plot_error_reductions_vs_batch_size(
        title,
        output_dir,
        reduction_dict,
        y_lim_right=[-0.01, 0.01]
    )

def test_make_gif():
    """ Test making a gif out of pre-existing image files """
    np.random.seed(1344)
    n_frames = 5
    x = np.linspace(0, 1, 50)
    t = np.linspace(0, 1, n_frames, endpoint=False)
    dir_name = os.path.join(output_dir, "Test make gif")
    image_path_list = []
    # First of all, save some image files to disk
    for t_i in t:
        y = np.sin(2 * np.pi * (x - t_i))
        plot_name = "Test make gif, t = %.2f" % t_i
        plotting.simple_plot(x, y, "x", "y", plot_name, dir_name, 1, "b-")
        output_filename = "%s.png" % plot_name
        output_path = os.path.join(dir_name, output_filename)
        image_path_list.append(output_path)
    # Now turn image files into a gif
    plotting.make_gif(
        output_name="Test make gif",
        output_dir=dir_name,
        input_path_list=image_path_list,
        duration=(1000 / n_frames)
    )
