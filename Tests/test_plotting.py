""" Module containing unit tests for the plotting module. The functions for
plotting activation and error functions are not tested here, but in the test
modules for the activation and error function modules. Similarly, the function
for plotting training curves is tested in the test module for the optimiser
module. """
import os
import numpy as np
from math import ceil
import pytest
import plotting, data, optimisers, models
from .util import get_output_dir, set_random_seed_from_args, get_random_network

# Get name of output directory, and create it if it doesn't exist
output_dir = get_output_dir("Plotting")

@pytest.mark.parametrize("output_dim", [1, 2, 3])
def test_plot_1D_regression(output_dim):
    """ Test plotting function for regression data with 1-dimensional inputs,
    and a variable number of outputs """
    set_random_seed_from_args("test_plot_1D_regression", output_dim)
    # Initialise data and model
    sin_data = data.Sinusoidal(
        input_dim=1,
        output_dim=output_dim,
        n_train=100,
        n_test=50,
        x_lo=0,
        x_hi=1,
    )
    model = get_random_network(
        input_dim=1,
        output_dim=output_dim,
        low=2,
        high=3,
        initialiser=models.initialisers.ConstantPreActivationStatistics(
            x_train=sin_data.x_train,
            y_train=sin_data.y_train,
        ),
    )
    # Call plotting function under test
    plotting.plot_1D_regression(
        plot_name="test_plot_1D_regression, output_dim=%i" % output_dim,
        dir_name=os.path.join(output_dir, "test_plot_1D_regression"),
        dataset=sin_data,
        model=model,
        output_dim=output_dim,
    )


@pytest.mark.parametrize("output_dim", [1, 2, 3])
def test_plot_1D_hidden_outputs(output_dim):
    """ Test plotting function for outputs from the hidden layers of a model
    1-dimensional inputs, and a variable number of outputs """
    # Set random seed and parameters
    set_random_seed_from_args("test_plot_1D_regression", output_dim)
    n_train = np.random.randint(50, 100)
    n_test  = np.random.randint(50, 100)
    n_pred  = np.random.randint(50, 100)
    # Initialise data and model
    sin_data = data.Sinusoidal(
        input_dim=1,
        output_dim=output_dim,
        n_train=n_train,
        n_test=n_test,
        x_lo=0,
        x_hi=1,
    )
    model = get_random_network(
        input_dim=1,
        output_dim=output_dim,
        low=2,
        high=6,
        initialiser=models.initialisers.ConstantPreActivationStatistics(
            x_train=sin_data.x_train,
            y_train=sin_data.y_train,
        ),
    )
    # Initialise result and column objects
    result = optimisers.Result(verbose=False, add_default_columns=False)
    prediction_column = optimisers.results.columns.Predictions(
        sin_data,
        n_points_per_dim=n_pred,
        store_hidden_layer_outputs=True,
    )
    result.add_column(prediction_column)
    # Call optimisation function
    optimisers.gradient_descent(
        model,
        sin_data,
        result=result,
        terminator=optimisers.Terminator(i_lim=1),
    )
    # Call plotting function under test
    plotting.plot_1D_hidden_outputs(
        plot_name="test_plot_1D_hidden_outputs, output_dim=%i" % output_dim,
        dir_name=os.path.join(output_dir, "test_plot_1D_hidden_outputs"),
        dataset=sin_data,
        x_pred=prediction_column.x_pred,
        hidden_output_list=prediction_column.hidden_outputs_dict[0],
        y_pred=prediction_column.predictions_dict[0],
        output_dim=output_dim,
    )


@pytest.mark.parametrize("seed, output_dim", [(1815, 1), (1743, 3)])
def test_plot_2D_regression(seed, output_dim):
    """ Test plotting function for regression data with 2 input dimensions and a
    variable number of output dimensions """
    np.random.seed(seed)
    input_dim = 2
    x_lo = -2
    x_hi = 2
    # Generate dataset and network
    sin_data = data.Sinusoidal(
        input_dim=input_dim,
        output_dim=output_dim,
        n_train=2000,
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
            y_train=sin_data.y_train,
        )
    )
    # Call plotting function under test
    plotting.plot_2D_regression(
        plot_name="Random predictions for 2D-%iD sinusoidal data" % output_dim,
        dir_name=output_dir,
        output_dim=output_dim,
        dataset=sin_data,
        model=model,
    )


def test_plot_2D_classification():
    """ Test plotting function for classification data with 2 input dimensions
    and a variable number of classes """
    # Set random seed and input and output dimensions
    set_random_seed_from_args("test_plot_2D_classification")
    input_dim = 2
    output_dim = np.random.randint(4, 7)
    # Generate dataset and network
    classification_data = data.MixtureOfGaussians(
        input_dim=input_dim,
        output_dim=output_dim,
        n_train=2000,
    )
    model = get_random_network(
        input_dim=input_dim,
        output_dim=output_dim,
        low=2,
        high=3,
        initialiser=models.initialisers.ConstantPreActivationStatistics(
            x_train=classification_data.x_train,
            y_train=classification_data.y_train,
        )
    )
    # Call plotting function under test
    plotting.plot_2D_classification(
        plot_name=(
            "test_plot_2D_classification for 2D-%iD classification data"
            % output_dim
        ),
        dir_name=output_dir,
        output_dim=output_dim,
        dataset=classification_data,
        model=model,
    )


def test_plot_binary_2D_classification():
    """ Test plotting function for binary classification data with 2 input
    dimensions """
    # Set random seed and input and output dimensions
    set_random_seed_from_args("test_plot_binary_2D_classification")
    input_dim = 2
    n_mixture_components = np.random.randint(4, 7)
    # Generate dataset and network
    classification_data = data.BinaryMixtureOfGaussians(
        input_dim=input_dim,
        n_mixture_components=n_mixture_components,
        n_train=2000,
    )
    model = get_random_network(
        input_dim=input_dim,
        output_dim=1,
        low=2,
        high=3,
        act_funcs=[models.activations.identity, models.activations.logistic],
        initialiser=models.initialisers.ConstantPreActivationStatistics(
            x_train=classification_data.x_train,
            y_train=classification_data.y_train,
        )
    )
    # Call plotting function under test
    plotting.plot_2D_binary_classification(
        plot_name=(
            "test_plot_binary_2D_classification for 2D classification data"
        ),
        dir_name=output_dir,
        dataset=classification_data,
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
        e_lims=None,
        n_iqr=1,
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

def test_plot_error_reductions_vs_batch_size_gif():
    """ Test function which plots a gif of the statistics for the reduction in
    the mean error in the test set after a single minimisation iteration, as a
    function of the batch size used for the iteration, where each frame in the
    gif represents a different iteration throughout the course of
    model-optimisation """
    # Set random seed and initialise network and dataset
    np.random.seed(102)
    n_train = 10
    n_its = 2
    model = get_random_network(input_dim=1, output_dim=1)
    sin_data = data.Sinusoidal(input_dim=1, output_dim=1, n_train=n_train)
    # Initialise Result, LineSearch and OptimalBatchSize column objects
    result = optimisers.Result(verbose=False)
    line_search = optimisers.LineSearch()
    gd_optimiser = optimisers.GradientDescent(line_search)
    columns = optimisers.results.columns
    optimal_batch_size_col = columns.OptimalBatchSize(
        gd_optimiser,
        sin_data.n_train,
        n_repeats=3,
        n_batch_sizes=3,
        min_batch_size=2
    )
    result.add_column(optimal_batch_size_col)
    # Call optimisation function
    gd_optimiser.optimise(
        model,
        sin_data,
        result=result,
        terminator=optimisers.Terminator(i_lim=n_its),
        evaluator=optimisers.Evaluator(i_interval=1),
    )
    # Call plotting function to make gif
    test_output_dir = os.path.join(
        output_dir,
        "Test plot_error_reductions_vs_batch_size_gif"
    )
    plotting.plot_error_reductions_vs_batch_size_gif(
        result,
        optimal_batch_size_col,
        test_output_dir,
        loop=None
    )

def test_plot_optimal_batch_sizes():
    """ Test function which plots the optimal batch size, rate of reduction of
    the mean test set error, and train and test error, against the current
    iteration throughout the course of model-optimisation """
    # Set random seed and initialise network and dataset
    np.random.seed(102)
    n_train = 10
    n_its = 2
    model = get_random_network(input_dim=1, output_dim=1)
    sin_data = data.Sinusoidal(input_dim=1, output_dim=1, n_train=n_train)
    # Initialise Result, LineSearch and OptimalBatchSize column objects
    result = optimisers.Result(verbose=False)
    line_search = optimisers.LineSearch()
    gd_optimiser = optimisers.GradientDescent(line_search)
    columns = optimisers.results.columns
    optimal_batch_size_col = columns.OptimalBatchSize(
        gd_optimiser,
        sin_data.n_train,
        n_repeats=3,
        n_batch_sizes=3,
        min_batch_size=2
    )
    result.add_column(optimal_batch_size_col)
    # Call optimisation function
    gd_optimiser.optimise(
        model,
        sin_data,
        result=result,
        terminator=optimisers.Terminator(i_lim=n_its),
        evaluator=optimisers.Evaluator(i_interval=1),
    )
    # Call plotting function
    plotting.plot_optimal_batch_sizes(
        "Test plot_optimal_batch_sizes",
        output_dir,
        result,
        optimal_batch_size_col
    )

@pytest.mark.parametrize(
    "dataset_type",
    [data.Sinusoidal, data.MixtureOfGaussians],
)
def test_plot_predictions_gif(dataset_type):
    """ Test function to make a gif of predictions formed by the model during
    training, for regression or classification """
    # Set random seed and initialise network and dataset
    set_random_seed_from_args("test_plot_predictions_gif", dataset_type)
    n_train = np.random.randint(10, 20)
    n_pred = np.random.randint(5, 10)
    n_its = 2
    if issubclass(dataset_type, data.Regression):
        input_dim = 1
        output_dim = 1
        dataset_kwargs = {"x_lo": -1, "x_hi": 1, "freq": 1}
    elif issubclass(dataset_type, data.Classification):
        input_dim = 2
        output_dim = 3
        dataset_kwargs = {}
    else:
        raise ValueError(
            "dataset_type %r must be a subclass of data.Regression or "
            "data.Classification" % dataset_type
        )
    dataset = dataset_type(
        input_dim=input_dim,
        output_dim=output_dim,
        n_train=n_train,
        **dataset_kwargs,
    )
    model = get_random_network(
        input_dim=input_dim,
        output_dim=output_dim,
        initialiser=models.initialisers.ConstantPreActivationStatistics(
            x_train=dataset.x_train,
            y_train=dataset.y_train,
        ),
    )
    # Initialise Result and Predictions column object
    result = optimisers.Result(verbose=False)
    pred_column = optimisers.results.columns.Predictions(dataset, n_pred)
    result.add_column(pred_column)
    # Call optimisation function
    optimisers.gradient_descent(
        model,
        dataset,
        result=result,
        terminator=optimisers.Terminator(i_lim=n_its),
        evaluator=optimisers.Evaluator(i_interval=1),
        line_search=optimisers.LineSearch(),
    )
    # Call plotting function
    plot_name = (
        "test_plot_predictions_gif, dataset_type = %s" % dataset_type.__name__
    )
    plotting.plot_predictions_gif(
        plot_name=plot_name,
        dir_name=os.path.join(output_dir, plot_name),
        result=result,
        prediction_column=pred_column,
        dataset=dataset,
        output_dim=output_dim,
        duration=1000,
    )
