""" TODO """
import numpy as np

class _Initialiser:
    """ TODO """
    def __init__(self, *args):
        raise NotImplementedError()

    def initialise_params(self, model, *args):
        raise NotImplementedError()

class ConstantParameterStatistics(_Initialiser):
    def __init__(
        self,
        weight_mean=0.0,
        weight_std=1.0,
        bias_mean=0.0,
        bias_std=1.0
    ):
        self.weight_mean    = weight_mean
        self.weight_std     = weight_std
        self.bias_mean      = bias_mean
        self.bias_std       = bias_std

    def initialise_params(self, model, *args):
        for layer in model.layers:
            layer.init_params(
                self.weight_mean,
                self.weight_std,
                self.bias_mean,
                self.bias_std
            )

class ConstantPreActivationStatistics(_Initialiser):
    def __init__(
        self,
        x_train,
        y_train,
        hidden_layer_mean=0.0,
        hidden_layer_std=1.0
    ):
        self.x_train            = x_train
        self.y_train            = y_train
        self.hidden_layer_mean  = hidden_layer_mean
        self.hidden_layer_std   = hidden_layer_std

    def initialise_params(self, model, *args):
        """
        Initialise parameters for the neural network.

        TODO: experiment with different standard deviations in each column of
        the weight matrix, as a function of the standard deviation of each row
        of the input matrix, so that each weighted row contributes an equal
        variance to the final output variance (rather than having constant
        statistics for each column of the weight matrix)
        """
        this_layer = model.layers[0]
        weights_std = (
            self.hidden_layer_std / np.sqrt(self.x_train.var(axis=1).sum())
        )
        weights = np.random.normal(
            0,
            weights_std,
            [this_layer.output_dim, this_layer.input_dim]
        )
        bias = self.hidden_layer_mean - (weights @ self.x_train).mean(axis=1)
        this_layer.init_params(weights, 0, bias, 0)
        this_layer.activate(self.x_train)

        for i in range(1, len(model.layers) - 1):
            layer_input = model.layers[i-1].output
            this_layer = model.layers[i]
            weights_std = (
                self.hidden_layer_std / np.sqrt(layer_input.var(axis=1).sum())
            )
            weights = np.random.normal(
                0,
                weights_std,
                [this_layer.output_dim, this_layer.input_dim]
            )
            bias = (
                self.hidden_layer_mean - (weights @ layer_input).mean(axis=1)
            )
            this_layer.init_params(weights, 0, bias, 0)
            this_layer.activate(layer_input)
        
        # TODO: add unit test for network with no hidden layers
        if len(model.layers) > 1:
            layer_input = model.layers[-2].output
            this_layer = model.layers[-1]
            weights_std = np.sqrt(np.divide(
                self.y_train.var(axis=1, keepdims=True),
                layer_input.var(axis=1).sum()
            ))
            weights = np.random.normal(
                0,
                weights_std,
                [this_layer.output_dim, this_layer.input_dim]
            )
            output_mean = self.y_train.mean(axis=1)
            pre_activation_mean = (weights @ layer_input).mean(axis=1)
            bias = output_mean - pre_activation_mean
            this_layer.init_params(weights, 0, bias, 0)


class FromModelFile(_Initialiser):
    def __init__(self, filename):
        raise NotImplementedError()

    def initialise_params(self, model, *args):
        raise NotImplementedError()
