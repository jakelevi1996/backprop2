"""
Module to contain the _Column class and its subclasses, which will be added as
attributes to Result objects, to represent and store the data in each column, as
well as being configurable with names, widths, and format specifiers for
printing

...TODO...
"""

import numpy as np
from _optimisers.batch import ConstantBatchSize as _ConstantBatchSize
from _optimisers.linesearch import LineSearch as _LineSearch
from _optimisers.terminator import Terminator as _Terminator
from _optimisers.evaluator import DoNotEvaluate as _DoNotEvaluate
from _optimisers.abstract_result import AbstractResult as _AbstractResult

class _Column:
    def __init__(self, name, format_spec, width=0):
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
        initialised with verbose=True.
        """
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
    """ TODO """
    def __init__(
        self,
        model,
        dataset,
        line_search,
        optimise_func,
        n_repeats=100,
        n_batch_sizes=30,
        min_batch_size=5,
        use_replacement=True,
        name="Optimal Batch size",
        format_spec="d",
    ):
        """ TODO 

        Is it necessary to initiate with model and dataset? Can get these in
        update method from kwargs
        """
        # Initialise results dictionaries
        self.reduction_dict_dict        = dict()
        self.mean_dict                  = dict()
        self.std_dict                   = dict()
        self.best_batch_dict            = dict()
        self.best_reduction_rate_dict   = dict()
        self.best_reduction_dict        = dict()
        # Store attributes that will be used in the update method
        self._model                     = model
        self._dataset                   = dataset
        self._reference_line_search     = line_search
        self._optimise_func             = optimise_func
        self._n_repeats                 = n_repeats
        self._use_replacement           = use_replacement
        self._terminator                = _Terminator(i_lim=1)
        self._evaluator                 = _DoNotEvaluate()
        self._result = _AbstractResult()
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
                dataset.n_train,
                n_batch_sizes
            )
            batch_size_list = set(int(b) for b in batch_size_list)
            if len(batch_size_list) < n_batch_sizes:
                n_batch_sizes += 1
            else:
                break
            if n_batch_sizes > dataset.n_train:
                break
        self.batch_size_list = np.array(sorted(batch_size_list))

        # Call parent initialiser to initialise format strings, value list etc
        super().__init__(name, format_spec)
    
    def update(self, kwargs):
        model = kwargs["model"]
        dataset = kwargs["dataset"]
        iteration = kwargs["iteration"]
        # Get parameters and current test error
        w_0 = self._model.get_parameter_vector().copy()
        self._model.forward_prop(self._dataset.x_test)
        E_0 = self._model.mean_error(self._dataset.y_test)
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
                    self._model,
                    self._dataset,
                    line_search=self._line_search,
                    terminator=self._terminator,
                    evaluator=self._evaluator,
                    result=self._result,
                    batch_getter=batch_getter,
                    display_summary=False,
                )
                # Calculate new error and add the reduction to the list
                self._model.forward_prop(self._dataset.x_test)
                E_new = self._model.mean_error(self._dataset.y_test)
                error_reduction = E_0 - E_new
                reduction_dict[batch_size].append(error_reduction)
                # Reset parameters
                self._model.set_parameter_vector(w_0.copy())

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
