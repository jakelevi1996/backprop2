""" TODO """

class _Initialiser:
    """ TODO """
    def __init__(self, *args):
        raise NotImplementedError()

    def initialise_params(self, model, *args):
        raise NotImplementedError()

class ConstantParameterStatistics(_Initialiser):
    def __init__(self, weight_std, bias_std):
        self.weight_std = weight_std
        self.bias_std = bias_std

    def initialise_params(self, model, *args):
        raise NotImplementedError()

class ConstantPreActivationStatistics(_Initialiser):
    def __init__(self, x_train, y_train, weight_std, bias_std):
        self.input_means    = x_train.mean(axis=-1)
        self.input_var      = x_train.var(axis=-1)
        self.output_means   = y_train.mean(axis=-1)
        self.output_var     = y_train.var(axis=-1)
        self.weight_std     = weight_std
        self.bias_std       = bias_std

    def initialise_params(self, model, *args):
        raise NotImplementedError()
