""" Package for creating DataSet objects, with input and output data, training
and test partitions, and methods for batching, saving, loading, and printing
the data.

TODO:
-   implement DataSet classes for CircleDataSet, SumOfGaussianCurvesDataSet,
    GaussianCurveDataSet
-   have a class/subclasses for generating input points (EG uniform, Gaussian,
    grid, etc), which is shared between all data classes?
"""
from data.dataset import DataSet
from data.regression import Regression
from data.classification import Classification, BinaryClassification
from data.sinusoidal import Sinusoidal
from data.mixture_of_gaussians import (
    MixtureOfGaussians,
    BinaryMixtureOfGaussians,
)
from data.xor import Xor
from data.disk import Disk
from data.circle import Circle
from data.square import Square
from data.guassian_curve import GaussianCurve, SumOfGaussianCurves
from data.task import TaskSet

# Create dictionary mapping name-strings to non-abstract dataset classes
dataset_class_dict = {
    dataset_class.__name__: dataset_class
    for dataset_class in [
        Sinusoidal,
        MixtureOfGaussians,
        BinaryMixtureOfGaussians,
        Xor,
        Disk,
        GaussianCurve,
        Square,
    ]
}
