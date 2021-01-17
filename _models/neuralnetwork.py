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
from _models import activations, errors, initialisers
from _models.layer import NeuralLayer

class NeuralNetwork():
    def __init__(
        self,
        input_dim=1,
        output_dim=1,
        num_hidden_units=None,
        act_funcs=None,
        error_func=None,
        initialiser=None,
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
        -   act_funcs: list of activation functions for each layer in the
            network. The length of this list can be different to the number of
            network layers: if this list is shorter than the number of layers,
            then the first activation function in the list is used for multiple
            layers; if this list is longer than the number of layers, then the
            first activation functions in the list are ignored. In either case,
            the final activation functions in this list will match the
            activation functions used in the final layers. Default is [logistic,
            identity], IE output is identity function, and all hidden layers
            have logistic activation functions
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
            act_funcs = [activations.logistic, activations.identity]
        if error_func is None:
            error_func = errors.sum_of_squares
        if initialiser is None:
            initialiser = initialisers.ConstantParameterStatistics()

        # Set network constants
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._num_units_list = list(num_hidden_units) + [output_dim]
        self._num_layers = len(self._num_units_list)
        self._error_func = error_func

        # Initialise layers
        self._init_layers(act_funcs)

        # Initialise parameters
        initialiser.initialise_params(self)

        # Initialise memory for the parameter and gradient vectors
        self.num_params = sum(
            layer.num_weights + layer.num_bias for layer in self.layers
        )
        self.param_vector = np.empty(self.num_params)
        self.grad_vector = np.empty(self.num_params)
    
    def _init_layers(self, act_func_list):
        """
        Initialise the layers of the neural network.
        """
        # Resize the list of activation functions
        act_func = lambda i: act_func_list[
            max(len(act_func_list) - self._num_layers + i, 0)
        ]

        # Initialise the input layer
        new_layer = NeuralLayer(
            self._num_units_list[0],
            self.input_dim,
            act_func(0),
        )
        self.layers = [new_layer]

        # Initialise the rest of the layers
        for i in range(1, self._num_layers):
            new_layer = NeuralLayer(
                self._num_units_list[i],
                self._num_units_list[i-1],
                act_func(i)
            )
            self.layers.append(new_layer)
    
    def glorot_initialiser(self, act_funcs):
        # TODO: Should initialisers have their own classes?
        raise NotImplementedError

    def forward_prop(self, x):
        """ Propogate an input forward through the network, and store and return
        the output from the network.

        Inputs:
        -   x: input to the neural network. Should be a numpy array with shape
            (input_dim, N_D)

        Outputs:
        -   layer_output: output from the final layer in the network, == the
            network output. Should be in a numpy array with shape (output_dim,
            N_D)
        """
        # Store the number of data points for use in other methods
        self.N_D = x.shape[1]
        # Calculate output from the first layer
        layer_output = self.layers[0].activate(x)
        # Calculate outputs from subsequent layers
        for layer in self.layers[1:]:
            layer_output = layer.activate(layer_output)
        # Store and return network output
        self.y = layer_output
        return layer_output

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
        """
        # Calculate error (can be reused in back_prop2 method)
        self._error = self._error_func.dEdy(self.y, target)
        # Calculate the output layer delta and gradients
        self.layers[-1].delta = np.multiply(
            self._error,
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

        NOTE: it is assumed that the forward_prop and back_prop methods have
        been called with the same arguments prior to calling this method (EG by
        the get_gradient_vector method)
        """
        # Calculate the output layer epsilon
        final_layer = self.layers[-1]
        final_layer.epsilon = np.einsum(
            "ijk,ik,jk->ijk",
            self._error_func.d2Edy2(self.y, target),
            self.layers[-1].act_func.dydx(self.layers[-1].pre_activation),
            self.layers[-1].act_func.dydx(self.layers[-1].pre_activation)
        )
        final_layer.epsilon[final_layer.diag_indices] += np.multiply(
            self._error,
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
        # Propagate data through the network to get gradients
        self.forward_prop(x)
        self.back_prop(x, target)
        # Initialise the pointer and iterate through the layers
        i = 0
        for layer in self.layers:
            # Set the weight gradients and update the pointer
            self.grad_vector[i:i+layer.num_weights] = (
                layer.w_grad.mean(axis=-1).ravel()
            )
            i += layer.num_weights
            # Set the bias gradients and update the pointer
            self.grad_vector[i:i+layer.num_bias] = (
                layer.b_grad.mean(axis=-1).ravel()
            )
            i += layer.num_bias
        
        return self.grad_vector

    def get_hessian_blocks(self, x, target, weight_ind_list, bias_ind_list):
        """
        Return the mean (across data points) of a block-diagonal approximation
        of the Hessian matrix of second derivatives of the error between the
        targets and the network's predictions based on x with respect to all of
        the parameters in the network, as a list of 2D matrices. Also return a
        list of the indices in the corresponding gradient vector (returned by
        the get_gradient_vector method) that each Hessian block corresponds to.

        Inputs:
        -   x: input to the neural network. Should be a numpy array with shape
            (input_dim, N_D)
        -   target: targets that the neural network is trying to predict. Should
            be a numpy array with shape (output_dim, N_D)
        -   weight_ind_list: 3D iterable. The outermost dimension refers to a
            layer in the network, and the number of elements in the outermost
            dimension should match the number of layers in the network. The next
            dimension refers to each block within the block-diagonal
            approximation of the Hessian, and the number of elements is equal to
            the number of blocks used to represent that layer. This iterable
            should be a np.ndarray. In the next (innermost) dimension, each
            element should be a non-negative integer, referring to which weight
            parameter corresponds to that index in that block of the Hessian
            approximation
        -   bias_ind_list: same as for weight_ind_list, but for bias parameters,
            instead of weights

        Outputs:
        -   hess_block_list: list of 2D matrices, containing blocks of the
            Hessian of the error function according to the specified input
            indices. The convention for the order of layers, weights and biases
            is consistent with that of the get_gradient_vector method
        -   hess_inds_list: list of 1D vectors of non-negative integers, each of
            which refers to an index in the gradient vector referred to by this
            element in the corresponding Hessian block (this is easier to
            explain and understand using an example; see the optimisers module)

        NOTE: it is assumed that the forward_prop and back_prop methods have
        been called with the same arguments prior to calling this method (EG by
        the get_gradient_vector method)

        TODO: could improve efficiency by calculating sum instead of mean, if
        the same was done with the first order gradients
        """
        # Calculate epsilons for each layer in the network
        self.back_prop2(x, target)

        # Initialise output lists and offset
        hess_block_list = []
        hess_inds_list = []
        offset = 0

        # Iterate through each layer in the network
        for i, layer in enumerate(self.layers):

            # Iterate through each block of weights in this layer
            for block_weight_inds in weight_ind_list[i]:
                # Calculate the Hessian block and update output lists
                hess_block = layer.calc_weight_gradients2(
                    block_weight_inds,
                    self.N_D
                )
                hess_block_list.append(hess_block.mean(axis=-1))
                hess_inds_list.append(block_weight_inds + offset)
                
            # Update the offset for hess_inds_list 
            offset += layer.num_weights
            
            # Iterate through each block of biases in this layer
            for block_bias_inds in bias_ind_list[i]:
                # Calculate the Hessian block and update output lists
                hess_block = layer.calc_bias_gradients2(block_bias_inds)
                hess_block_list.append(hess_block.mean(axis=-1))
                hess_inds_list.append(block_bias_inds + offset)
            
            # Update the offset for hess_inds_list 
            offset += layer.num_bias
        
        return hess_block_list, hess_inds_list


    def get_dbs_metric(self):
        """
        TODO: Write docstring

        NOTE: it is assumed that the forward_prop and back_prop methods have
        been called prior to calling this method (EG by calling the
        get_gradient_vector method)
        """
        return min(layer.get_dbs_metric() for layer in self.layers)

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

    def mean_error(self, t):
        """ Calculate the mean (across all data points) of the error between the
        given targets and the network's predictions for the most recent set of
        inputs that were propagated through the network.

        Inputs:
        -   t: targets that the neural network is trying to predict. Should be a
            numpy array with shape (output_dim, N_D), where N_D is the number of
            data points (size of dimension 1) of the most recent set of inputs
            that were propagated through the network

        Outputs:
        -   e: mean error across all data points, as a numpy float64 scalar
        """
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

    def __call__(self, x, t=None, w=None):
        """ Wrapper for the forward_prop method, and optionally also the
        set_parameter_vector and mean_error methods. If parameters are provided,
        then they are first set as the model parameters. Next, the inputs are
        propagated through the network. Finally, if targets are provided, then
        the mean error between the network output and the targets is calculated
        and returned. Otherwise the network output is returned.

        Inputs:
        -   x: inputs to be propagated through the network
        -   t (optional): targets, used to calculate mean error
        -   w (optional): new parameters to apply to the model
        """
        # Update parameter vector, if new parameters are provided
        if w is not None:
            self.set_parameter_vector(w)
        # Propagate inputs through the network
        y = self.forward_prop(x)
        # If targets are provided, then calculate and return the mean error
        if t is not None:
            return self.mean_error(t)
        # Otherwise return the network output
        else:
            return y

def load_network(filename, dir_name):
    raise NotImplementedError()
    return NeuralNetwork(initialiser=initialisers.FromModelFile(filename))
