import os
import numpy as np
from _optimisers import columns
from _optimisers.abstract_result import AbstractResult
from _optimisers.timer import Timer

# Initialise default column types added to a Result object
DEFAULT_COLUMN_TYPES = [
    columns.Iteration,
    columns.Time,
    columns.TrainError,
    columns.TestError
]

class Result(AbstractResult):
    """
    Class to store the results of optimisation in a single object which can be
    passed directly to plotting and/or analysis functions. Also contains methods
    for updating and displaying results.

    NOTE: saving and loading of Result objects is currently deprecated, because
    due to updating this module with Column objects, implementing saving and
    loading of Result objects in npz files would currently require more effort
    to develop than is worthwhile. Alternative are to manually extract
    value_lists from each column and save those in a npz file, or just pickle
    the whole Result object. TODO: implement saving and loading of Result
    objects
    """
    def __init__(
        self,
        name=None,
        verbose=True,
        file=None,
        add_default_columns=True
    ):
        """ Store the name of the experiment (which is useful later when
        displaying results), verbosity, output file, and initialise list of
        columns, and dictionary mapping types of columns to their instances. If
        specified by the input argument, then add default columns to this Result
        object. """
        self.name = name if (name is not None) else "Unnamed experiment"
        self.file = file
        self.verbose = verbose
        self._column_list = list()
        self._column_dict = dict()
        self._iteration = 0
        self.begun = False
        if add_default_columns:
            self._add_default_columns()

    def _add_default_columns(self):
        for col in DEFAULT_COLUMN_TYPES:
            self.add_column(col())

    def add_column(self, column):
        if column.name in self._column_dict:
            raise ValueError("A column with this name has already been added")

        self._column_list.append(column)
        self._column_dict[type(column)] = column

    def get_iteration_number(self):
        """ Return the current iteration number, as an integer """
        return self._iteration
    
    def get_values(self, column_type):
        """ Given the type of column, return the list of values for the column
        with the matching type.

        Raises KeyError if this Result object does not have a Column with a
        matching type. """
        return self._column_dict[column_type].value_list
    
    def get_column_name(self, column_type):
        """ Given the type of column, return the name of the column with the
        matching type.

        Raises KeyError if this Result object does not have a Column with a
        matching type. """
        return self._column_dict[column_type].name
    
    def begin(self):
        """ Display column headers for the columns in this result (if this
        result object is verbose, and these column headers have not already been
        displayed by calling this method previously), and if this result does
        not have a timer object, then add a timer object, and tell it to begin.

        If a specific timer object is to be added to this Result class, which
        can be done using the set_timer method (inherited from the TimedObject
        class via the AbstractResult class), then the set_timer method should be
        called before this method (Result.begin), in order to avoid
        unnecessarily initialising an extra timer object """
        if self.verbose:
            self._display_headers()
        
        if not self.has_timer():
            timer = Timer()
            self.set_timer(timer)
            timer.begin()
        
        self.begun = True
    
    def update(self, **kwargs):
        """ Update all the columns in this Result object with new values.
        Depending on the columns used by this object, certain keyword arguments
        will be required; if the default columns are used, then the keyword
        arguments model, dataset, and iteration will be required. The begin
        method must be called before the update method, otherwise an
        AttributeError is raised. """
        kwargs["time"] = self.time_elapsed()
        self._iteration = kwargs.get("iteration")

        for col in self._column_list:
            col.update(kwargs)

        if self.verbose:
            self._display_last()
    
    def _display_headers(self):
        title_list = [col.title_str for col in self._column_list]
        print("\nPerforming test \"{}\"...".format(self.name),  file=self.file)
        print(" | ".join(title_list),                           file=self.file)
        print(" | ".join("-" * len(t) for t in title_list),     file=self.file)

    def _display_last(self):
        """
        Display the results of the last time the update method was called.
        Raises IndexError if update has not been called on this object before 
        """
        print(
            " | ".join(col.get_value_str() for col in self._column_list),
            file=self.file
        )

    def display_summary(self, n_iters):
        t_total = self.time_elapsed()
        t_mean = t_total / n_iters
        print(
            "-" * 50,
            "{:30} = {}".format("Test name", self.name),
            "{:30} = {:,.4f} s".format("Total time", t_total),
            "{:30} = {:,}".format("Total iterations", n_iters),
            "{:30} = {:.4f} ms".format(
                "Average time per iteration",
                1e3 * t_mean
            ),
            "{:30} = {:,.1f}".format(
                "Average iterations per second",
                1 / t_mean
            ),
            sep="\n",
            end="\n\n",
            file=self.file
        )
    
    def __repr__(self):
        return "Result({})".format(repr(self.name))
