import os
from time import perf_counter
import numpy as np
from _optimisers import columns

# Define list of attributes which are saved and loaded by the Result class
attr_name_list = [
    "train_errors",
    "test_errors",
    "times",
    "iters",
    "step_size",
    "start_time",
]

class Result():
    """
    Class to store the results of optimisation in a single object which can be
    passed directly to plotting and/or analysis functions. Also contains methods
    for updating and displaying results

    TODO:
    - Make this class configurable, so columns such as step-size and |x| are
    optional, and the column width and format spec for each column is
    configurable
    """
    def __init__(
        self,
        name=None,
        verbose=True,
        file=None,
        add_default_columns=True
    ):
        """
        Store the name of the experiment (which is useful later when displaying
        results), display table headers, initialise lists for objective function
        evaluations and the time and iteration number for each evaluation, and
        record the start time for the results list.
        """
        self.name = name if (name is not None) else "Unnamed experiment"
        self.file = file
        self.verbose = verbose
        self._column_list = list()
        self._column_dict = dict()
        if add_default_columns:
            self._add_default_columns()

    def _add_default_columns(self):
        for col in [
            columns.Iteration,
            columns.Time,
            columns.TrainError,
            columns.TestError
        ]:
            self.add_column(col())

    def add_column(self, column):
        if column.name in self._column_dict:
            raise ValueError("A column with this name has already been added")

        self._column_list.append(column)
        self._column_dict[column.name] = column
    
    def get_values(self, name):
        """
        Given the input string name, return the list of values for the column
        with the matching name.
        """
        return self._column_dict[name].value_list
    
    def begin(self):
        if self.verbose:
            self.display_headers()
        
        self.start_time = perf_counter()
    
    def time_elapsed(self):
        return perf_counter() - self.start_time
    
    def update(self, **kwargs):
        kwargs["time"] = self.time_elapsed()

        for col in self._column_list:
            col.update(kwargs)

        if self.verbose:
            self.display_last()
    
    def display_headers(self):
        title_list = [col.title_str for col in self._column_list]
        print("\nPerforming test \"{}\"...".format(self.name),  file=self.file)
        print(" | ".join(title_list),                           file=self.file)
        print(" | ".join("-" * len(t) for t in title_list),     file=self.file)

    def display_last(self):
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
    
    def save(self, filename, dir_name="."):
        """
        Save all of the relevant attributes of the Result object in a numpy file

        TODO: this needs to be updated in several ways, EG to save self.name,
        and to save the relevant attributes for each column. Maybe each column
        should have a method to return a specially formatted dictionary, which
        can be saved using np.savez? Maybe this method should wrap a module-wide
        results.save functions, which is capable of saving lists of experiments?
        How to deal with lists of Result objects with duplicate names? Could
        store the number of repeats of each name in the savez dictionary?
        """
        path = os.path.abspath(os.path.join(dir_name, filename))
        np.savez(
            path,
            **{a: getattr(self, a) for a in attr_name_list}
        )

    def load(self, filename, dir_name="."):
        """
        Load the result attributes from the specified filename, replacing the
        values of the attributes in the object which is calling this method, and
        making sure that each attribute has the correct type

        TODO: this method needs to be updated; see comments above
        """
        path = os.path.abspath(os.path.join(dir_name, filename))
        with np.load(path) as data:
            for a in attr_name_list:
                a_type = type(getattr(self, a))
                a_val = data[a]
                setattr(self, a, a_type(a_val))
    
    def __repr__(self):
        return "Result({})".format(repr(self.name))

def load(filename, dir_name="."):
    """ Load a results file. Wrapper for the Result.load method """
    r = Result()
    r.load(filename, dir_name)
    return r
