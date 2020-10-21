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
        init_weight_std = np.sqrt(1.0 / self.x_train.var(axis=1).sum())
        init_bias = -(init_weight_std @ self.x_train).mean(axis=1)
        model.layers[0].init_params(0, init_weight_std, init_bias, 0)
        model.layers[0].activate(self.x_train)

        for i in range(1, len(model.layers) - 1):
            layer_input = model.layers[i-1].output
            init_weight_std = np.sqrt(1.0 / layer_input.var(axis=1).sum())
            init_bias = -(init_weight_std @ layer_input).mean(axis=1)
            model.layers[i].init_params(0, init_weight_std, init_bias, 0)
            model.layers[i].activate(self.x_train)
        
        # TODO: will this work if there is only 1 layer?
        layer_input = model.layers[-2].output
        init_weight_std = np.sqrt(
            self.y_train.var(axis=1) / layer_input.var(axis=1).sum()
        )
        init_bias = -(init_weight_std @ layer_input).mean(axis=1)
        model.layers[-1].init_params(0, init_weight_std, init_bias, 0)


class FromModelFile(_Initialiser):
    def __init__(self, filename):
        raise NotImplementedError()

    def initialise_params(self, model, *args):
        raise NotImplementedError()
