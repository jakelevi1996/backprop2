import os
from time import perf_counter
import numpy as np

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
    def __init__(self, name=None, verbose=True, file=None):
        """
        Store the name of the experiment (which is useful later when displaying
        results), display table headers, initialise lists for objective function
        evaluations and the time and iteration number for each evaluation, and
        record the start time for the results list.
        """
        self.name = name if (name is not None) else "Unnamed experiment"
        self.file = file
        if verbose:
            self.display_headers()
        self.verbose = verbose

        self.train_errors   = []
        self.test_errors    = []
        self.times          = []
        self.iters          = []
        self.step_size      = []
        # TODO: DBS criterion
        self.begin()
    
    def begin(self):
        self.start_time = perf_counter()
    
    def time_elapsed(self):
        return perf_counter() - self.start_time
    
    def update(self, model, dataset, i, s):
        t = self.time_elapsed()
        e_train = model.mean_error(dataset.y_train, dataset.x_train)
        e_test  = model.mean_error(dataset.y_test, dataset.x_test)
        self.train_errors.append(e_train)
        self.test_errors.append(e_test)
        self.times.append(t)
        self.iters.append(i)
        self.step_size.append(s)

        if self.verbose:
            self.display_last()
    
    def display_headers(self):
        # num_fields, field_width = 3, 10
        print("\nPerforming test \"{}\"...".format(self.name), file=self.file)
        print(
            "{:9} | {:8} | {:11} | {:11} | {:10}".format(
                "Iteration",
                "Time (s)",
                "Train error",
                "Test error",
                "Step size",
            ),
            file=self.file
        )
        print(" | ".join("-" * i for i in [9, 8, 11, 11, 10]), file=self.file)

    def display_last(self):
        """
        Display the results of the last time the update method was called.
        Raises IndexError if update has not been called on this object before 
        """
        print(
            "{:9d} | {:8.3f} | {:11.5f} | {:11.5f} | {:10.4f}".format(
                self.iters[-1],
                self.times[-1],
                self.train_errors[-1],
                self.test_errors[-1],
                self.step_size[-1]
            ),
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
