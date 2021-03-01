""" Module containing the _Column class and its subclasses, which can be added
to Result objects, to store and represent the data in each column, as well as
being configurable with names, widths, and format specifiers for printing.
Columns will typically be added to a Result object before optimisation starts
(by default and/or by initialising a specific type of column and passing it to
the Result.add_column method), and data from the column (for example the mean
test-set error during the current iteration) will be displayed every time the
model is evaluated. """

import numpy as np
from _optimisers.batch import ConstantBatchSize as _ConstantBatchSize
from _optimisers.linesearch import LineSearch as _LineSearch
from _optimisers.terminator import Terminator as _Terminator
from _optimisers.evaluator import DoNotEvaluate as _DoNotEvaluate
from _optimisers.abstract_result import AbstractResult as _AbstractResult

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
        """ Given the dictionary kwargs passed from minimise to Result.update to
        this method, extract the appropriate value for this column from kwargs,
        and add it to the end of this object's internal list of values. This
        method will be overriden by all subclasses of _Column. """
        raise NotImplementedError

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
        train_error = model.mean_error(dataset.y_train)
        self.value_list.append(train_error)

class TestError(_Column):
    def __init__(self, name="Test error", format_spec=".5f"):
        super().__init__(name, format_spec)
    
    def update(self, kwargs):
        dataset = kwargs["dataset"]
        model = kwargs["model"]
        model.forward_prop(dataset.x_test)
        test_error = model.mean_error(dataset.y_test)
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
    plotting.plot_optimal_batch_sizes """
    def __init__(
        self,
        max_batch_size,
        optimise_func,
        line_search,
        n_repeats=100,
        n_batch_sizes=30,
        min_batch_size=5,
        use_replacement=True,
        name="Optimal Batch size",
    ):
        """ Initialise an OptimalBatchSize object.

        Inputs:
        -   max_batch_size: the maximum batch size to test when looking for the
            optimal batch size. Should be no bigger than the size of the
            training set
        -   optimise_func: function to call in order to perform one iteration of
            optimisation, for each repeat and for each different batch size to
            test, in order to determine the optimal batch size. Should accept a
            model and a dataset as the first arguments, as well as the following
            keyword arguments: line_search, terminator, evaluator, result,
            batch_getter, display_summary. [1]
        -   line_search: instance of _optimisers.linesearch.LineSearch, or None.
            Should match the LineSearch object which is being used during
            optimisation. [1]
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

        [1] TODO: if the minimise function was turned into a private method of a
        Minimiser class, and each minimisation function was converted into a
        subclass of the Minimiser object which called this common method,
        wrapped by a function, then this column could simply be passed that
        instance of the specific Minimiser subclass, instead of an optimisation
        function which might have to specially initiated as a lambda expression
        to contain the correct parameters.
        """
        # Initialise results dictionaries
        self.reduction_dict_dict        = dict()
        self.mean_dict                  = dict()
        self.std_dict                   = dict()
        self.best_batch_dict            = dict()
        self.best_reduction_rate_dict   = dict()
        self.best_reduction_dict        = dict()
        # Store attributes that will be used in the update method
        self._reference_line_search     = line_search
        self._optimise_func             = optimise_func
        self._n_repeats                 = n_repeats
        self._use_replacement           = use_replacement
        self._terminator                = _Terminator(i_lim=1)
        self._evaluator                 = _DoNotEvaluate()
        self._result                    = _AbstractResult()
        if line_search is None:
            self._line_search = None
            self._reference_line_search = None
        else:
            self._line_search = _LineSearch(
                alpha=line_search.alpha,
                beta=line_search.beta,
                max_its=line_search.max_its
            )
            self._reference_line_search = line_search

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
        model = kwargs["model"]
        dataset = kwargs["dataset"]
        iteration = kwargs["iteration"]
        # Get parameters and current test error
        w_0 = model.get_parameter_vector().copy()
        model.forward_prop(dataset.x_test)
        E_0 = model.mean_error(dataset.y_test)
        reduction_dict = dict()
        # Iterate through batch sizes
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
                if self._line_search is not None:
                    self._line_search.s = self._reference_line_search.s
                self._optimise_func(
                    model,
                    dataset,
                    line_search=self._line_search,
                    terminator=self._terminator,
                    evaluator=self._evaluator,
                    result=self._result,
                    batch_getter=batch_getter,
                    display_summary=False,
                )
                # Calculate new error and add the reduction to the list
                model.forward_prop(dataset.x_test)
                E_new = model.mean_error(dataset.y_test)
                error_reduction = E_0 - E_new
                reduction_dict[batch_size].append(error_reduction)
                # Reset parameters
                model.set_parameter_vector(w_0.copy())

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


# Create dictionary mapping names to _Column subclasses, for saving/loading
column_names_dict = {col.__name__: col for col in _Column.__subclasses__()}
