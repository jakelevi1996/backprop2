"""
Module to contain neural network classes, which themselves contain methods for
initialisation, evaluation, and gradient calculations.

Notation convention: it is assumed that each neural network has L hidden layers
and 1 output layer, with N_0 units in the first (input) hidden layer, N_1 units
in the second hidden layer, ..., N_L-1 units in the last hidden layer, and N_L
units in the output layer. Each model can accept multidimensional inputs and
outputs; the dimension of the input data is input_dim, and the dimension of the
output data is output_dim == N_L == the number of units in the output layer.
Each neural network can accept multiple input points simultaneously; when inputs
are fed to a neural network, it is assumed that N_D input points are fed
simultaneously in a numpy array with shape (input_dim, N_D) (IE the first array
dimension refers to the dimension within each input point, and the second array
dimension refers to a particular input point)*. When such an input is fed into
the neural network, the output will be a numpy array with shape (output_dim,
N_D).

*This convention is chosen to simplify matrix multiplication during
forward-propagation and back-propagation; it is possible that in future the
array shape dimensions will be swapped, and the matrix multiplication operations
will be carried out with np.einsum operations with subscripts [0, 1] and [2, 1]
(whereas conventional matrix multiplication is equivalent to np.einsum with
subscripts [0, 1] and [1, 2])

TODO: within the whole repository, it might be more efficient to initialise
arrays and then use the `out` optional argument of the numpy functions and
methods to save repeated mallocs and frees (which I assume is similar to what
happens under the hood).
"""

import numpy as np
import activations as a, errors as e
from layer import NeuralLayer

class NeuralNetwork():
    def __init__(
        self,
        input_dim=1,
        output_dim=1,
        num_hidden_units=None,
        act_funcs=None,
        error_func=None,
        weight_std=1.0,
        bias_std=1.0,
        filename=None
    ):
        """
        __init__: initialise the network-constants and the weights, biases and
        activation functions for each layer of the neural network

        Inputs:
        -   input_dim: dimension of the input to the neural network
        -   output_dim: dimension of the output from the neural network, == the
            number of units in the final/output layer of the network
        -   num_hidden_units: list of integers, describing the number of units
            in each hidden layer of the network (input layer first); the length
            of the list is equal to the number of hidden layers. Default value
            is [10] (IE a single hidden layer with 10 units)
        -   weight_std: standard deviation for initialisation of weights
        -   bias_std: standard deviation for initialisation of biases
        -   act_funcs: list of activation functions for each layer in the
            network. The length of this list can be different to the number of
            network layers: if this list is shorter than the number of layers,
            then the first activation function in the list is used for multiple
            layers; if this list is longer than the number of layers, then the
            first activation functions in the list are ignored. In either case,
            the final activation functions in this list will match the
            activation functions used in the final layers. Default is
            [a.Logistic(), a.Identity()], IE output is identity function, and
            all hidden layers have logistic activation functions
        -   error_func: error function which (along with its gradients) is used
            to calculate gradients for the network parameters

        Outputs: initialised network

        TODO: instead of having num_hidden_units and act_funcs arguments as
        lists, should add layers one by one with an add_layer argument, and
        initialise weights once all the layers have been added?
        """
        # Set default number of hidden units and activation and error functions
        if num_hidden_units is None:
            num_hidden_units = [10]
        if act_funcs is None:
            act_funcs = [a.Logistic(), a.Identity()]
        if error_func is None:
            error_func = e.SumOfSquares()

        # Set network constants
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._num_units_list = list(num_hidden_units) + [output_dim]
        self._num_layers = len(self._num_units_list)
        self._error_func = error_func

        # Load or initialise weights
        if filename is not None:
            self.load_model(filename)
        else:
            num_params = self.gaussian_initialiser(
                act_funcs, weight_std, bias_std
            )

        # Initialise memory for the parameter and gradient vectors
        self.param_vector = np.empty(num_params)
        self.grad_vector = np.empty(num_params)
    
    def gaussian_initialiser(self, act_func_list, weight_std, bias_std):
        """
        gaussian_initialiser: initialise the layers of the neural network, using
        a Gaussian distribution with zero mean, a single standard deviation for
        all weights, and a single standard deviation for all biases. Also
        initialise the memory for the parameter and gradient vectors.

        TODO: if this becomes an external function/method, it can accept as
        additional arguments self._num_units_list (from which it can calculate
        the number of layers) and self.input_dim, and return the list of layers
        the number of parameters
        """
        # Resize the list of activation functions
        act_func = lambda i: act_func_list[
            max(len(act_func_list) - self._num_layers + i, 0)
        ]
        # Initialise the input layer
        new_layer = NeuralLayer(
            self._num_units_list[0], self.input_dim, act_func(0),
            weight_std, bias_std
        )
        self.layers = [new_layer]
        num_params = new_layer.num_weights + new_layer.num_bias
        # Initialise the rest of the layers
        for i in range(1, self._num_layers):
            new_layer = NeuralLayer(
                self._num_units_list[i], self._num_units_list[i-1],
                act_func(i), weight_std, bias_std
            )
            self.layers.append(new_layer)
            num_params += (new_layer.num_weights + new_layer.num_bias)
        
        return num_params
    
    def glorot_initialiser(self, act_funcs):
        # TODO: Should initialisers have their own classes?
        raise NotImplementedError

    def forward_prop(self, x):
        """
        forward_prop: propogate an input forward through the network, and store
        and return the output from the network.

        Inputs:
        -   x: input to the neural network. Should be a numpy array with shape
            (input_dim, N_D)

        Outputs:
        -   layer_output: output from the final layer in the network, == the
            network output. Should be in a numpy array with shape (output_dim,
            N_D)
        """
        # Calculate output from the first layer
        layer_output = self.layers[0].activate(x)
        # Calculate outputs from subsequent layers
        for layer in self.layers[1:]:
            layer_output = layer.activate(layer_output)
        # Store and return network output
        self.y = layer_output
        return self.y

    def back_prop(self, x, target):
        """
        Propogate an input forward through the network to form a prediction, and
        then propogate the derivatives of the error between the predictions and
        the target backwards, in order to calculate the derivative of the error
        function with respect to the parameters of the network.

        Inputs:
        -   x: input to the neural network. Should be a numpy array with shape
            (input_dim, N_D)
        -   target: targets that the neural network is trying to predict. Should
            be a numpy array with shape (output_dim, N_D)

        Outputs:
        -   self.y (not returned): network output, in a numpy array with shape
            (output_dim, N_D)
        -   gradients (not returned): the gradients for all parameters in the
            network are calculated and stored in self.layers[i].w_grad and
            self.layers[i].b_grad, ready to be extracted using the
            get_gradient_vector method

        TODO: call forward prop separately (outside of this method)            
        """
        # Perform forward propagation to calculate activations
        self.forward_prop(x)
        # Calculate the output layer delta and gradients
        self.layers[-1].delta = np.multiply(
            self._error_func.dEdy(self.y, target),
            self.layers[-1].act_func.dydx(self.layers[-1].pre_activation)
        )
        self.layers[-1].calc_gradients()
        # Calculate deltas and gradients for hidden layers
        for i in reversed(range(self._num_layers - 1)):
            self.layers[i].backprop(self.layers[i + 1])
            self.layers[i].calc_gradients()

    def back_prop2(self, x, target):
        """
        Calculate epsilon (second partial derivate of error with respect to
        pre-activations) for each layer in the network. Requires back_prop
        method has already been called, in order to calculate gradients and
        layer inputs and outputs.

        TODO:
        -   call forward prop and back prop separately (outside of this method)
        -   Reuse self._error_func.dEdy(self.y, target) instead of recalculating
        """
        # Perform forward propagation and back propagation to calculate 1st
        # order gradients
        self.back_prop(x, target)
        # Calculate the output layer epsilon
        final_layer = self.layers[-1]
        final_layer.epsilon = np.einsum(
            "ijk,ik,jk->ijk",
            self._error_func.d2Edy2(self.y, target),
            self.layers[-1].act_func.dydx(self.layers[-1].pre_activation),
            self.layers[-1].act_func.dydx(self.layers[-1].pre_activation)
        )
        final_layer.epsilon[final_layer.diag_indices] += np.multiply(
            self._error_func.dEdy(self.y, target),
            final_layer.act_func.d2ydx2(final_layer.pre_activation)
        )
        # Calculate epsilons for hidden layers
        for i in reversed(range(self._num_layers - 1)):
            self.layers[i].backprop2(self.layers[i + 1])

    def get_parameter_vector(self):
        """
        get_parameter_vector: return all the parameters in the network as a long
        1D vector. NB if the returned parameter vector is intended to be used
        later to restore the model state, then a copy should be made using
        get_parameter_vector().copy() (otherwise the memory locations pointed to
        by the output from this function may be modified later)

        Inputs: None

        Outputs:
        -   parameter_vector: numpy array with shape (num_params, ) containing
            all of the parameters in the network, in the order: layer 0 weights,
            layer 0 bias, layer 1 weights, layer 1 bias, ..., final layer
            weights, final layer bias
        """
        # Initialise the pointer and iterate through the layers
        i = 0
        for layer in self.layers:
            # Set the weights and update the pointer
            self.param_vector[i:i+layer.num_weights] = layer.weights.ravel()
            i += layer.num_weights
            # Set the biases and update the pointer
            self.param_vector[i:i+layer.num_bias] = layer.bias.ravel()
            i += layer.num_bias
        
        return self.param_vector

    def get_gradient_vector(self, x, target):
        """
        get_gradient_vector: return the mean (across data points) of the
        gradients (of the error between the targets and the network's
        predictions based on x) with respect to all of the parameters in the
        network, as a long 1D vector

        Inputs:
        -   x: input to the neural network. Should be a numpy array with shape
            (input_dim, N_D)
        -   target: targets that the neural network is trying to predict. Should
            be a numpy array with shape (output_dim, N_D)

        Outputs:
        -   gradient_vector: numpy array with shape (num_params, ) containing
            the gradients of the error function with respect to all of the
            parameters in the network, in the order: layer 0 weights, layer 0
            bias, layer 1 weights, layer 1 bias, ..., final layer weights, final
            layer bias (same convention as the get_parameter_vector method)
        """
        self.back_prop(x, target)
        # Initialise the pointer and iterate through the layers
        i = 0
        for layer in self.layers:
            # Set the weight gradients and update the pointer
            self.grad_vector[i:i+layer.num_weights] = (
                layer.w_grad.mean(axis=-1).ravel()
            )
            i += layer.num_weights
            # Set the biase gradients and update the pointer
            self.grad_vector[i:i+layer.num_bias] = (
                layer.b_grad.mean(axis=-1).ravel()
            )
            i += layer.num_bias
        
        return self.grad_vector

    def get_hessian(self, block_list): raise NotImplementedError

    def set_parameter_vector(self, new_parameters):
        """
        set_parameter_vector: set the values of all of the network parameters

        Inputs:
        -   new_parameters: numpy array with shape (num_params, ) containing all
            of the new parameters for the network, in the order: layer 0 weights,
            layer 0 bias, layer 1 weights, layer 1 bias, ..., final layer
            weights, final layer bias (same convention as the
            get_parameter_vector method)
        """
        # Initialise the pointer and iterate through the layers
        i = 0
        for layer in self.layers:
            # Set the weights and update the pointer
            layer.weights = new_parameters[i:i+layer.num_weights].reshape(
                layer.output_dim, layer.input_dim
            )
            i += layer.num_weights
            # Set the biases and update the pointer
            layer.bias = new_parameters[i:i+layer.num_bias].reshape(
                layer.output_dim, 1
            )
            i += layer.num_bias

    def mean_error(self, t, x=None):
        """
        mean_error: calculate the mean (across all data points) of the error
        between the given targets and the network's predictions. If a set of
        inputs is provided then the network predictions are calculated using
        these inputs; otherwise the most recently calculated network predictions
        are used, in which case the targets provided to this function must have
        the same number of data points (axis 1 of the numpy array).

        Inputs:
        -   t: targets that the neural network is trying to predict. Should be a
            numpy array with shape (output_dim, N_D)
        -   x (optional): input to the neural network, in a numpy array with
            shape (input_dim, N_D). If x == None (default value), then the most
            recently calculated network predictions are used, in which case the
            targets provided to this function must have the same number of data
            points (axis 1 of the numpy array)

        TODO: input order should be x, t, with no default for x. Could also wrap
        this into the __call__ method, with optional arguments t and w; if w is
        given, then call self.set_parameter_vector before calling
        self.mean_error

        Outputs:
        -   e: mean error across all data points, as a numpy float64 scalar
        """
        if x is not None: self.forward_prop(x)
        else: assert self.y.shape[1] == t.shape[1]

        return self._error_func(self.y, t).mean()

    def save_model(self, filename, filename_prepend_timestamp=True):
        # Save params, self._num_units_list, and identities of activation
        # functions as np.ndarrays in an npz file, and return the filename
        # params = self.get_parameter_vector()
        raise NotImplementedError
    
    def load_model(self, filename):
        # params = np.load(filename)
        # self.set_parameter_vector(params)
        raise NotImplementedError

    def print_weights(self, file=None):
        for i, layer in enumerate(self.layers):
            print(
                "Layer {}/{}:".format(i + 1, self._num_layers),
                "Weights:", layer.weights,
                "Biases:", layer.bias,
                "-" * 50, sep="\n", file=file
            )

    def print_grads(self, file=None):
        for i, layer in enumerate(self.layers):
            print(
                "Layer {}/{}:".format(i + 1, self._num_layers),
                "Weight gradients:", layer.w_grad,
                "Bias gradients:", layer.b_grad,
                "-" * 50, sep="\n", file=file
            )

    def __call__(self, x):
        """
        Wrapper for the forward_prop method. TODO: add optional arguments t and
        w, which are used to calculate mean error, and optionally set the
        parameter vector first. Could be used to tidy up the optimisers module.

        def __call__(self, x, t=None, w=None):
            if w is not None: self.set_parameter_vector(w)
            if t is None: return return self.forward_prop(x)
            else: return self.mean_error(t, x) # swap round t and x
        """
        return self.forward_prop(x)
