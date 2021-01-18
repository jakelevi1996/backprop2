"""
Module to contain the _Column class and its subclasses, which will be added as
attributes to Result objects, to represent and store the data in each column, as
well as being configurable with names, widths, and format specifiers for
printing

...TODO...
"""

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
        """
        Return the last value with which this _Column object was updated,
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
        """
        Initialise a column to use for step sizes. This column must be
        initialised with the LineSearch object that will be used during
        minimisation with the Result that this column will belong to.
        """
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

# Create dictionary mapping names to _Column subclasses, for saving/loading
column_names_dict = {col.__name__: col for col in _Column.__subclasses__()}
