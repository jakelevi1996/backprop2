import numpy as np
from _optimisers.minimise import minimise, Result

class NewtonStepCalculator():
    # TODO: make it possible to specify that the block indices should be
    # regenerated on every iteration
    def __init__(self, model, max_block_size, max_step, learning_rate):
        self.model = model

        # Get random indices for block-diagonalisation of weights in each layer
        self.weight_inds = [
            np.array_split(
                np.random.permutation(layer.num_weights),
                np.ceil(layer.num_weights / max_block_size)
            ) for layer in model.layers
        ]

        # Get random indices for block-diagonalisation of biases in each layer
        self.bias_inds = [
            np.array_split(
                np.random.permutation(layer.num_bias),
                np.ceil(layer.num_bias / max_block_size)
            ) for layer in model.layers
        ]

        self.delta = np.empty(model.num_params)

        self.max_step = max_step
        self.learning_rate = learning_rate
    
    def get_step(self, model, dataset):
        # Get gradient vector
        dEdw = model.get_gradient_vector(dataset.x_train, dataset.y_train)
        # Get Hessian blocks
        (hess_block_list, hess_inds_list) = model.get_hessian_blocks(
            dataset.x_train, dataset.y_train, self.weight_inds, self.bias_inds
        )
        # Iterate through each Hessian block
        for hess_block, hess_inds in zip(hess_block_list, hess_inds_list):
            # Rotate gradient into eigenbasis of Hessian
            evals, evecs = np.linalg.eigh(hess_block)
            grad_rot = np.matmul(evecs.T, dEdw[hess_inds])
            # Take a Newton step in directions in which this step is not too big
            step_rot = np.where(
                (self.max_step * np.abs(evals)) > np.abs(grad_rot),
                -grad_rot / np.abs(evals),
                -self.learning_rate * grad_rot
            )
            # Rotate gradient back into original coordinate system and return
            self.delta[hess_inds] = np.matmul(evecs, step_rot)
        
        return self.delta, dEdw


def generalised_newton(
    model,
    dataset,
    learning_rate=1e-1,
    max_block_size=7,
    max_step=1,
    result=None,
    **kwargs
):
    newton_step_calculator = NewtonStepCalculator(
        model,
        max_block_size,
        max_step,
        learning_rate
    )

    get_step = lambda model, dataset: newton_step_calculator.get_step(
        model,
        dataset
    )

    if result is None:
        result = Result("Generalised Newton")

    result = minimise(model, dataset, get_step, result=result, **kwargs)

    return result
