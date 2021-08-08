from data.dataset import DataSet

class Classification(DataSet):
    """ Class for classification datasets. Outputs are one-hot integer
    matrices, with self.output_dim number of rows (this is equal to the number
    of classes), and each column refers to a different data point. Each column
    has one value equal to 1, referring to which class that data point belongs
    to, and the rest of the values are equal to zero. This class is used as a
    parent class for specific classification datasets. """

class BinaryClassification(Classification):
    """ Class for binary classification datasets. Outputs are binary matrices
    with a single row, where a value of 0 or 1 determines the class of the data
    point, and each column (IE each element) refers to a different data point.
    This class is used as a parent class for specific binary classification
    datasets. """
