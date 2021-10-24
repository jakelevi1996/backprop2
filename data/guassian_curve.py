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
    x_affine_transformed = np.dot(input_scale, x - input_offset)
    y_pre_affine_transformation = (
        np.exp(-np.square(x_affine_transformed).sum(axis=0))
    )
    noisy_offset = np.random.normal(
        loc=output_offset,
        scale=noise_std,
        size=[output_dim, x.shape[1]],
    )
    y = (y_pre_affine_transformation * output_scale) + noisy_offset
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
        Regression.__init__(self)
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
        self.train.x = np.random.uniform(
            x_lo,
            x_hi,
            size=[input_dim, self.train.n],
        )
        self.test.x  = np.random.uniform(
            x_lo,
            x_hi,
            size=[input_dim, self.test.n],
        )
        self.train.y = _noisy_gaussian(
            self.train.x,
            input_offset,
            input_scale,
            output_offset,
            output_scale,
            noise_std,
            output_dim,
        )
        self.test.y = _noisy_gaussian(
            self.test.x,
            input_offset,
            input_scale,
            output_offset,
            output_scale,
            noise_std,
            output_dim,
        )
