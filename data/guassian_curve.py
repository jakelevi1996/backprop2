import numpy as np
from data.regression import Regression

def _noisy_gaussian(
    x,
    input_offset,
    input_scale,
    output_offset,
    output_scale,
    noise_std,
    output_dim,
):
    noisy_offset = np.random.normal(
        loc=output_offset,
        scale=noise_std,
        size=[output_dim, x.shape[1]],
    )
    y = (
        (
            np.exp(-np.square(input_scale @ (x - input_offset)).sum(axis=0))
            * output_scale
        )
        + noisy_offset
    )
    return y

class SumOfGaussianCurves(Regression):
    pass

class GaussianCurve(Regression):
    def __init__(
        self,
        input_dim=1,
        output_dim=1,
        n_train=None,
        n_test=None,
        x_lo=-2,
        x_hi=2,
        noise_std=0.1,
        input_offset=None,
        input_scale=None,
        output_offset=None,
        output_scale=None,
    ):
        """ Initialise a GaussianCurve object """
        # Set shape constants and unspecified parameters
        self.set_shape_constants(input_dim, output_dim, n_train, n_test)
        if input_offset is None:
            input_offset = np.random.normal(size=[input_dim, 1])
        if input_scale is None:
            input_scale = np.random.normal(size=[input_dim, input_dim])
        if output_offset is None:
            output_offset = np.random.normal(size=[output_dim, 1])
        if output_scale is None:
            output_scale = np.random.normal(scale=5, size=[output_dim, 1])
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
        self.y_train = _noisy_gaussian(
            self.x_train,
            input_offset,
            input_scale,
            output_offset,
            output_scale,
            noise_std,
            output_dim,
        )
        self.y_test = _noisy_gaussian(
            self.x_test,
            input_offset,
            input_scale,
            output_offset,
            output_scale,
            noise_std,
            output_dim,
        )
