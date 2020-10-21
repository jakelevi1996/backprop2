""" TODO """

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
        raise NotImplementedError()

class FromModelFile(_Initialiser):
    def __init__(self, filename):
        raise NotImplementedError()

    def initialise_params(self, model, *args):
        raise NotImplementedError()
