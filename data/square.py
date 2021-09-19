import numpy as np
from data.regression import Regression

def _square(x, output_dim):
    in_square_element_wise = np.logical_and(x > -1, x < 1, dtype=int)
    in_square = in_square_element_wise.prod(axis=0, keepdims=True, dtype=float)
    y = np.tile(in_square, [output_dim, 1])
    return y

class Square(Regression):
    def __init__(
        self,
        input_dim=2,
        output_dim=1,
        n_train=None,
        n_test=None,
        x_lo=-2,
        x_hi=2,
    ):
        """ Initialise a dataset for regression which consists of a square """
        # Set shape constants
        self.set_shape_constants(input_dim, output_dim, n_train, n_test)
        # Generate input/output training and test data
        self.x_train = np.random.uniform(
            x_lo,
            x_hi,
            size=[input_dim, self.n_train]
        )
        self.x_test  = np.random.uniform(
            x_lo,
            x_hi,
            size=[input_dim, self.n_test]
        )
        self.y_train    = _square(self.x_train, output_dim)
        self.y_test     = _square(self.x_test,  output_dim)
