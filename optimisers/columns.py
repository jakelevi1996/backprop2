""" Module containing the _Column class and its subclasses, which can be added
to Result objects, to store and represent the data in each column, as well as
being configurable with names, widths, and format specifiers for printing.
Columns will typically be added to a Result object before optimisation starts
(by default and/or by initialising a specific type of column and passing it to
the Result.add_column method), and data from the column (for example the mean
test-set error during the current iteration) will be displayed every time the
model is evaluated. """

import numpy as np
from scipy.stats import norm
from optimisers.batch import ConstantBatchSize as _ConstantBatchSize
from optimisers.linesearch import LineSearch as _LineSearch
from optimisers.terminator import Terminator as _Terminator
from optimisers.evaluator import DoNotEvaluate as _DoNotEvaluate
from optimisers.abstract_result import AbstractResult as _AbstractResult
from optimisers import smooth

class _Column:
    """ Abstract column class, to be subclassed. All subclasses should override
    the update method as a minimum, and preferably also the __init__ method,
    setting defaults for the column name and format spec, before calling this
    class' __init__ method """
    def __init__(self, name, format_spec, width=0):
        """ Initialise the attributes for this _Column object, including its
        name, title string (printed as the column heading at the start of
        minimisation), value list, and format string (used to format values of
        this column object before printing them during optimisation) """
        self.name = name
        width = max(width, len(name))
        self.title_str = "{{:{}s}}".format(width).format(name)
        self.value_list = []
        self._value_fmt_str = "{{:{}{}}}".format(width, format_spec)

    def update(self, kwargs):
        """ Given the dictionary kwargs passed from AbstractOptimiser.optimise
        to Result.update to this method, extract the appropriate value for this
        column from kwargs, and add it to the end of this object's internal list
        of values. This method will be overriden by all subclasses of _Column.
        """
        raise NotImplementedError()

    def get_value_str(self):
        """ Return the last value with which this _Column object was updated,
        formatted as a string according to self._value_fmt_str, which depends on
        the width and format_spec with which this object was initialised. The
        string is used by the Result.update method, if the Result was
        initialised with verbose=True. """
        return self._value_fmt_str.format(self.value_list[-1])

class Iteration(_Column):
    def __init__(self, name="Iteration", format_spec="d"):
        super().__init__(name, format_spec)

    def update(self, kwargs):
        self.value_list.append(kwargs["iteration"])

class Time(_Column):
    def __init__(self, name="Time (s)", format_spec=".3f"):
        super().__init__(name, format_spec)

    def update(self, kwargs):
        self.value_list.append(kwargs["time"])

class TrainError(_Column):
    def __init__(self, name="Train error", format_spec=".5f"):
        super().__init__(name, format_spec)

    def update(self, kwargs):
        dataset = kwargs["dataset"]
        model = kwargs["model"]
        model.forward_prop(dataset.x_train)
        train_error = model.reconstruction_error(dataset.y_train).mean()
        self.value_list.append(train_error)

class TestError(_Column):
    def __init__(self, name="Test error", format_spec=".5f"):
        super().__init__(name, format_spec)

    def update(self, kwargs):
        dataset = kwargs["dataset"]
        model = kwargs["model"]
        model.forward_prop(dataset.x_test)
        test_error = model.reconstruction_error(dataset.y_test).mean()
        self.value_list.append(test_error)

class StepSize(_Column):
    def __init__(self, line_search, name="Step Size", format_spec=".4f"):
        """ Initialise a column to use for step sizes. This column must be
        initialised with the LineSearch object that will be used during
        minimisation """
        self.line_search = line_search
        super().__init__(name, format_spec)

    def update(self, kwargs):
        self.value_list.append(self.line_search.s)

class DbsMetric(_Column):
    def __init__(self, name="DBS metric", format_spec=".4f"):
        super().__init__(name, format_spec)

    def update(self, kwargs):
        self.value_list.append(kwargs["model"].get_dbs_metric())

class BatchSize(_Column):
    def __init__(self, batch_getter, name="Batch size", format_spec="d"):
        self.batch_getter = batch_getter
        super().__init__(name, format_spec)

    def update(self, kwargs):
        self.value_list.append(int(self.batch_getter.batch_size))

class OptimalBatchSize(_Column):
    """ This column is used to approximate the optimal batch size during
    training, where the optimal batch size is considered to be the one which
    maximises the ratio of [the reduction in the mean error of the test set
    after a single minimisation iteration] over [the batch size used for the
    iteration].

    The motivation for calculating this information is that typically over the
    course of minimisation, we want to reduce the mean test set error as much as
    possible as fast as possible; using a large batch size will give more
    reliably large reductions, however will also take longer for each iteration.

    The information calculated by this column object when the model is evaluated
    is stored in dictionary attributes, which can be plotted EG by passing this
    object to plotting.plot_error_reductions_vs_batch_size_gif or
    plotting.plot_optimal_batch_sizes.

    See the test_optimal_batch_size_column function in the Tests/test_columns.py
    module for a usage example of this class """
    def __init__(
        self,
        optimiser,
        max_batch_size,
        n_repeats=100,
        n_batch_sizes=30,
        min_batch_size=5,
        use_replacement=True,
        name="Optimal Batch size",
    ):
        """ Initialise an OptimalBatchSize object.

        Inputs:
        -   optimiser: instance of AbstractOptimiser which is being used to
            optimise the model when this column is being used to approximate the
            optimal batch size
        -   max_batch_size: the maximum batch size to test when looking for the
            optimal batch size. Should be no bigger than the size of the
            training set
        -   n_repeats: number of repeats to perform of each batch size during a
            given iteration. The statistics of the reduction in the test set
            error for a given test set tend to be quite highly stochastic, so a
            large number of repeats is recommended. The default is 100
        -   n_batch_sizes: number of batch sizes to test. A larger number will
            take more time each time the model is evaluated, but will give a
            more accurate approximation of the optimal batch size, as well as a
            more fine-grained view of how the statistics of test-set error
            reduction depend on the batch size
        -   min_batch_size: the maximum batch size to test when looking for the
            optimal batch size. Should be >= 1
        -   use_replacement: whether or not to use replacement when sampling
            batches from the training set. Default is True
        -   name: name of the column object, used as the column heading when
            evaluating the model
        """
        # Initialise results dictionaries
        self.reduction_dict_dict        = dict()
        self.mean_dict                  = dict()
        self.std_dict                   = dict()
        self.best_batch_dict            = dict()
        self.best_reduction_rate_dict   = dict()
        self.best_reduction_dict        = dict()
        # Store attributes that will be used in the update method
        self._optimiser                 = optimiser
        self._using_line_search         = (optimiser.line_search is not None)
        self._n_repeats                 = n_repeats
        self._use_replacement           = use_replacement
        self._terminator                = _Terminator(i_lim=1)
        self._evaluator                 = _DoNotEvaluate()
        self._result                    = _AbstractResult()

        # Initialise list of unique integer batch sizes of the correct length
        while True:
            batch_size_list = np.linspace(
                min_batch_size,
                max_batch_size,
                n_batch_sizes
            )
            batch_size_list = set(int(b) for b in batch_size_list)
            if len(batch_size_list) < n_batch_sizes:
                n_batch_sizes += 1
            else:
                break
            if n_batch_sizes > max_batch_size:
                break
        self.batch_size_list = np.array(sorted(batch_size_list))

        # Call parent initialiser to initialise format strings, value list etc
        super().__init__(name, format_spec="d")

    def update(self, kwargs):
        """ Find the current optimal batch size for this model and dataset
        during the current iteration, by testing each batch size for the
        specified number of repeats with which this object was initialised. """
        # Get the model, dataset, and iteration number
        model = kwargs["model"]
        dataset = kwargs["dataset"]
        iteration = kwargs["iteration"]
        # Get parameters and current test error
        w_0 = model.get_parameter_vector().copy()
        if self._using_line_search:
            s_0 = self._optimiser.line_search.s
        model.forward_prop(dataset.x_test)
        E_0 = model.mean_total_error(dataset.y_test)
        # Iterate through batch sizes
        reduction_dict = dict()
        for batch_size in self.batch_size_list:
            # Initialise results list and batch-getter
            reduction_dict[batch_size] = []
            batch_getter = _ConstantBatchSize(
                int(batch_size),
                replace=self._use_replacement
            )
            # Iterate through repeats of the batch size
            for _ in range(self._n_repeats):
                # Perform one iteration of gradient descent
                self._optimiser.optimise(
                    model,
                    dataset,
                    terminator=self._terminator,
                    evaluator=self._evaluator,
                    result=self._result,
                    batch_getter=batch_getter,
                    display_summary=False,
                )
                # Calculate new error and add the reduction to the list
                model.forward_prop(dataset.x_test)
                E_new = model.mean_total_error(dataset.y_test)
                error_reduction = E_0 - E_new
                reduction_dict[batch_size].append(error_reduction)
                # Reset parameters
                model.set_parameter_vector(w_0.copy())
                if self._using_line_search:
                    self._optimiser.line_search.s = s_0

        # Calculate arrays of mean and standard deviation reductions
        mean = np.array([
            np.mean(reduction_dict[batch_size])
            for batch_size in self.batch_size_list
        ])
        std = np.array([
            np.std(reduction_dict[batch_size])
            for batch_size in self.batch_size_list
        ])
        # Find optimal batch size and reduction
        mean_over_batch = mean / self.batch_size_list
        best_reduction_rate = max(mean_over_batch)
        i_best = list(mean_over_batch).index(best_reduction_rate)
        best_batch_size = self.batch_size_list[i_best]
        best_reduction = mean[i_best]
        # Store results for this iteration in dictionaries
        self.reduction_dict_dict[iteration]         = reduction_dict
        self.mean_dict[iteration]                   = mean
        self.std_dict[iteration]                    = std
        self.best_batch_dict[iteration]             = best_batch_size
        self.best_reduction_rate_dict[iteration]    = best_reduction_rate
        self.best_reduction_dict[iteration]         = best_reduction
        # Update the results list
        self.value_list.append(best_batch_size)

class Predictions(_Column):
    """ This column is for storing predictions of the model during training.
    Once training is complete, this object can be passed to the
    plot_predictions_gif object to plot a gif of the predictions evolving during
    training.

    See the test_predictions_column function in the Tests/test_columns.py module
    for a usage example of this class. """
    def __init__(
        self,
        dataset,
        n_points_per_dim=50,
        store_hidden_layer_outputs=False,
        store_hidden_layer_preactivations=False,
        name="Predictions",
    ):
        """ Initialise this Predictions column.

        Inputs:
        -   dataset: dataset that the model is about to be trained on. The
            training set from this dataset is used to calculate the upper and
            lower limits of the prediction inputs used by this object
        -   n_points_per_dim: the number of unique prediction inputs to use in
            each input dimension. The actual prediction inputs will be created
            as a mesh over each input dimension, so the total number of
            prediction inputs will by n_points_per_dim ** dataset.input_dim
        -   store_hidden_layer_outputs: if True, then store not only the
            prediction outputs, but also the outputs from each hidden layer in
            the model (this can be used to plot the evolution of each hidden
            unit during training)
        -   store_hidden_layer_preactivations: if True, and
            store_hidden_layer_outputs is True, then the preactivations for
            each hidden layer are stored (IE the output from the linear
            transformation of the layer's input, used as input to the layer's
            activation function), instead of the outputs from the hidden
            layer's activation function. If store_hidden_layer_outputs is
            False, then this argument has no effect. This argument might be
            useful EG if plotting the outputs from multiple hidden layers, when
            all activations are limited to a specific range such as [0, 1], to
            make it easier to tell the difference between the outputs from
            different units
        -   name: name of the column object, used as the column heading when
            evaluating the model
        """
        super().__init__(name, "s")
        self.n_points_per_dim = n_points_per_dim
        self.predictions_dict = dict()
        self.hidden_outputs_dict = dict()
        self.x_unique = np.linspace(
            dataset.x_test.min(axis=1),
            dataset.x_test.max(axis=1),
            num=n_points_per_dim,
            axis=1,
        )
        x_mesh = np.meshgrid(*self.x_unique)
        self.x_pred = np.stack([xx_i.ravel() for xx_i in x_mesh], axis=0)
        self.value_list.append("Yes")
        self.store_hidden_layer_outputs = store_hidden_layer_outputs
        self.store_preactivations = store_hidden_layer_preactivations

    def update(self, kwargs):
        """ Store the predictions of the model on this object's internal
        prediction inputs, and optionally also the activations of each hidden
        unit in the network.

        This method is called by the Result.update method, which is called by
        the AbstractOptimiser.optimise method, which is wrapped by various
        optimiser classes/functions. """
        # Get the model and iteration number
        model       = kwargs["model"]
        iteration   = kwargs["iteration"]
        # Calculate and store the predictions of the model
        y_pred = model(self.x_pred)
        self.predictions_dict[iteration] = y_pred
        # If specified, store the hidden layer outputs or preactivations
        if self.store_hidden_layer_outputs:
            if self.store_preactivations:
                self.hidden_outputs_dict[iteration] = [
                    layer.pre_activation.copy()
                    for layer in model.layers[:-1]
                ]
            else:
                self.hidden_outputs_dict[iteration] = [
                    layer.output.copy()
                    for layer in model.layers[:-1]
                ]

class TestSetImprovementProbabilitySimple(_Column):
    """ This column is for comparing the probability of improving the error in
    the test set between successive iterations, by comparing the mean test-set
    error during the previous iteration that this column was updated with the
    mean and standard deviation of the test-set error during the current
    iteration.

    Simple (in the name of this class) refers to not also using the standard
    deviation of the test-set error during the previous iteration that this
    column was updated.

    Note that if the error function for the given model and data-set
    combination is not bounded below (EG separable binary mixture of Gaussians
    with a single-layer model with logistic activation function and no
    regularisation), then it may be inappropriate to use the values calculated
    by this column as a metric for convergence, because the probability of
    improving may not be expected to converge towards 50% """

    def __init__(
        self,
        model,
        dataset,
        name="p(improve)_test_set",
        format_spec=".5f",
        smoother=None,
        use_cdf=False,
    ):
        super().__init__(name, format_spec)
        model.forward_prop(dataset.x_test)
        self.prev_mean_test_error = (
            model.reconstruction_error(dataset.y_test).mean()
        )
        self._smoother = smoother
        self._use_cdf = use_cdf

    def update(self, kwargs):
        dataset = kwargs["dataset"]
        model = kwargs["model"]
        model.forward_prop(dataset.x_test)
        error = model.reconstruction_error(dataset.y_test)
        mean_test_error = error.mean()
        std_test_error = error.std()
        p_improve = (
            (self.prev_mean_test_error - mean_test_error) / std_test_error
        )
        if self._use_cdf:
            p_improve = norm.cdf(p_improve)
        if self._smoother is not None:
            p_improve = self._smoother.smooth(p_improve)
        self.value_list.append(p_improve)
        self.prev_mean_test_error = mean_test_error


class BatchImprovementProbability(_Column):
    def __init__(
        self,
        dynamic_terminator,
        name="p(improve)_batch",
        format_spec=".4f",
    ):
        """ Initialise a column to track the most recent probability of
        improvement on each consecutive batch, according to a DynamicTerminator
        object. A DynamicTerminator must be used during optimisation, and this
        column must be initialised with that DynamicTerminator object """
        self._dynamic_terminator = dynamic_terminator
        super().__init__(name, format_spec)

    def update(self, kwargs):
        self.value_list.append(self._dynamic_terminator.p_improve)

# Create dictionary mapping names to _Column subclasses, for saving/loading
column_names_dict = {col.__name__: col for col in _Column.__subclasses__()}
