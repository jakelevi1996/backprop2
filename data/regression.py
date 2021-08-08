from data.dataset import DataSet

class Regression(DataSet):
    """ Class for regression datasets. Outputs are continuous matrices with
    self.output_dim numbers of rows, and each column refers to a different data
    point. This class is used as a parent class for specific regression
    datasets. """
