"""
Module to contain the _Column class and its subclasses, which will be added as
attributes to Result objects, to represent and store the data in each column, as
well as being configurable with names, widths, and format specifiers for
printing

...TODO...
"""

class _Column:
    def __init__(self, name, width, format_spec, title_name=None):
        self.name = name
        title_name = title_name if (title_name is not None) else name
        self.title_str = "{{:{}s}}".format(width).format(title_name)
        self.value_list = []
        self._value_fmt_str = "{{:{}{}}}".format(width, format_spec)
    
    def update(self, kwargs):
        """
        Given the dictionary kwargs passed from minimise to Result.update to
        this method, extract the appropriate value for this column from kwargs,
        and add it to the end of this object's internal list of values. This
        method will be overriden by some subclasses of _Column but not others.
        """
        self.value_list.append(kwargs[self.name])

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
    def __init__(
        self,
        name="iteration",
        width=9,
        format_spec="d",
        title_name="Iteration"
    ):
        super().__init__(name, width, format_spec, title_name)

class Time(_Column):
    def __init__(
        self,
        name="time",
        width=8,
        format_spec=".3f",
        title_name="Time (s)"
    ):
        super().__init__(name, width, format_spec, title_name)

class TrainError(_Column):
    def __init__(
        self,
        name="train_error",
        width=11,
        format_spec=".5f",
        title_name="Train error"
    ):
        super().__init__(name, width, format_spec, title_name)
    
    def update(self, kwargs):
        dataset = kwargs["dataset"]
        self.value_list.append(
            kwargs["model"].mean_error(dataset.y_train, dataset.x_train)
        )

class TestError(_Column):
    def __init__(
        self,
        name="test_error",
        width=11,
        format_spec=".5f",
        title_name="Test error"
    ):
        super().__init__(name, width, format_spec, title_name)
    
    def update(self, kwargs):
        dataset = kwargs["dataset"]
        self.value_list.append(
            kwargs["model"].mean_error(dataset.y_test, dataset.x_test)
        )

class StepSize(_Column):
    """
    TODO: this class could be initialised with the LineSearch object that is
    passed to the minimise wrapper, store it as an attribute, and get the s
    attribute directly from this object, rather than
    """
    def __init__(
        self,
        name="step_size",
        width=10,
        format_spec=".4f",
        title_name="Step Size"
    ):
        super().__init__(name, width, format_spec, title_name)
    
    def update(self, kwargs):
        self.value_list.append(kwargs["line_search"].s)

class DbsMetric(_Column):
    def __init__(
        self,
        name="dbs_metric",
        width=9,
        format_spec=".4f",
        title_name="DBS metric"
    ):
        super().__init__(name, width, format_spec, title_name)
        raise NotImplementedError()

class BatchSize(_Column):
    def __init__(
        self,
        name="batch_size",
        width=9,
        format_spec="d",
        title_name="Batch size"
    ):
        super().__init__(name, width, format_spec, title_name)
        raise NotImplementedError()
