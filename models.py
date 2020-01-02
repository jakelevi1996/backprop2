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

class Model():
    """
    Model: interface class for models that can be optimised by the optimiser
    class.

    TODO: update function arguments
    """
    def __call__(self, x):
        raise NotImplementedError

    def get_parameter_vector(self):
        raise NotImplementedError

    def get_gradient_vector(self):
        raise NotImplementedError

    def get_hessian(self):
        raise NotImplementedError

    def set_parameter_vector(self, new_parameters):
        raise NotImplementedError

def resize_list(old_list, new_len):
    """
    resize_list: add or remove elements from an existing list in order to change
    its length. This function is used in the NeuralNetwork initialisation method
    in case the length of the list of activation functions is different to the
    number of network layers. If the old list is too long, then remove the
    beginning of the list; if the old list is too short, then repeat the first
    element in the list. The original list is not modified in-place. NB: this
    function could be implemented with the following one-liner (which is more
    concise but less readable):

    return ((new_len - len(old_list)) * [old_list[0]]) + old_list[-new_len:]

    Inputs:
    -   old_list: the list that will be resized
    -   new_len: the length that the new list will have

    Outputs:
    -   list which has length new_len and the same final elements as old_list
    """
    if new_len < len(old_list):
        # Make the list shorter by removing the beginning
        return old_list[-new_len:]
    elif new_len > len(old_list):
        # Make the list longer by repeating the first element
        return ((new_len - len(old_list)) * [old_list[0]]) + old_list
    else:
        # Return original list
        return old_list

def check_reduce(*arg_list):
    arg_set = set(arg_list)
    assert len(arg_set) == 1
    return arg_set.pop()


class NeuralLayer():
    def __init__(self, num_units, num_inputs, act_func, weight_std, bias_std):
        """
        __init__: initialise the constants and parameters for a neural network
        layer. This method is called during NeuralNetwork.__init__()
        """
        # Set layer constants
        self.input_dim = num_inputs
        self.output_dim = num_units
        self.act_func = act_func
        
        # Randomly initialise parameters
        self.weights = np.random.normal(0, weight_std, [num_units, num_inputs])
        self.bias = np.random.normal(0, bias_std, [num_units, 1])
    
    def activate(self, input):
        """
        activate: calculate the pre-activation and output of this layer as a
        function of the input, and store the input for subsequent gradient
        calculations.

        Inputs:
        -   input: input to the layer. Should be a numpy array with shape
            (input_dim, N_D)

        Outputs:
        -   self.output: output from the layer, in a numpy array with shape
            (output_dim, N_D)
        -   self.pre_activation (not returned): result of linear transformation
            being applied to the input, before nonlinearity is applied to give
            the output, in a numpy array with shape (output_dim, N_D)
        """
        self.input = input
        self.pre_activation = np.matmul(self.weights, input) + self.bias
        self.output = self.act_func(self.pre_activation)
        return self.output
    
    def backprop(self, next_layer):
        """
        backprop: calculate delta (gradient of the error function with respect
        to the pre-activations) for the current layer of the network, using the
        delta and weights of the next layer in the network.

        Inputs:
        -   next_layer: the next layer in the network, as an instance of
            NeuralLayer. Must have pre-calculated delta for that layer, and its
            delta and weights must be the correct shape
        """
        self.delta = np.einsum(
            "jk,ji,ik->ik", next_layer.delta, next_layer.weights,
            self.act_func.dydx(self.pre_activation)
        )
    
    def calc_gradients(self):
        """
        calc_gradients: calculate the gradients of the error function with
        respect to the bias and weights in this layer, using the delta for this
        layer, which must have already been calculated using
        self.backprop(next_layer).

        TODO: test whether optimising the path gives a speed-up, and if so then
        calculate path in advance using np.einsum_path during initialisation.

        NB if the deltas and inputs are transposed, then the einsum subscripts
        should be "ij,ik->ijk"
        """
        self.b_grad = self.delta
        self.w_grad = np.einsum("ik,jk->ijk", self.delta, self.input)


class NeuralNetwork(Model):
    def __init__(
        self, input_dim=1, output_dim=1, num_hidden_units=[10],
        act_funcs=[a.Logistic(), a.Identity()], error_func=e.SumOfSquares(),
        weight_std=1.0, bias_std=1.0, filename=None
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
            network layers; see the resize_list function for a description of
            how the list is resized to match the number of network layers.
            Default is [a.Logistic(), a.Identity()], IE output is identity
            function, and all hidden layers have logistic activation functions
        -   error_func: error function which (along with its gradients) is used
            to calculate gradients for the network parameters

        Outputs: None
        """
        # Set network constants
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_units_list = num_hidden_units + [output_dim]
        self.num_layers = len(self.num_units_list)
        self.error_func = error_func

        # Load or initialise weights
        if filename is not None:
            self.load_model(filename)
        else:
            self.gaussian_initialiser(act_funcs, weight_std, bias_std)
    
    def gaussian_initialiser(self, act_funcs, weight_std, bias_std):
        """
        gaussian_initialiser: initialise the layers of the neural network, using
        a Gaussian distribution with zero mean, a single standard deviation for
        all weights, and a single standard deviation for all biases
        """
        # Resize the list of activation functions
        act_funcs = resize_list(act_funcs, self.num_layers)
        # Initialise the input layer
        self.layers = [NeuralLayer(
            self.num_units_list[0], self.input_dim, act_funcs[0],
            weight_std, bias_std
        )]
        # Initialise the rest of the layers
        for (num_units, num_inputs, act_func) in zip(
            self.num_units_list[1:], self.num_units_list[:-1], act_funcs[1:]
        ):
            self.layers.append(NeuralLayer(
                num_units, num_inputs, act_func, weight_std, bias_std
            ))
    
    def glorot_initialiser(self, act_funcs):
        # TODO: Should initialisers have their own classes?
        raise NotImplementedError

    def forward_prop(self, x):
        """
        forward_prop: propogate an input forward through the network, and return
        the output from the network.

        Inputs:
        -   x: input to the neural network. Should be a numpy array with shape
            (input_dim, N_D)

        Outputs:
        -   layer_output: output from the final layer in the network, == the
            network output. Should be in a numpy array with shape
            (output_dim, N_D)
        """
        # Calculate output from the first layer
        layer_output = self.layers[0].activate(x)
        # Calculate outputs from subsequent layers
        for layer in self.layers[1:]:
            layer_output = layer.activate(layer_output)
        # Return network output
        return layer_output

    def __call__(self, x):
        """
        __call__: wrapper for the forward_prop method
        """
        return self.forward_prop(x)

    def back_prop(self, x, target):
        """
        back_prop: propogate an input forward through the network to form a
        prediction, and then propogate the derivatives of the error between the
        predictions and the target backwards, in order to calculate the
        derivative of the error function with respect to the parameters of the
        network.

        Inputs:
        -   x: input to the neural network. Should be a numpy array with shape
            (input_dim, N_D)
        -   target: targets that the neural network is trying to predict. Should
            be a numpy array with shape (output_dim, N_D)

        Outputs:
        -   y: network output, in a numpy array with shape (output_dim, N_D)
        -   gradients (not returned): the gradients for all parameters in the
            network are calculated and stored in self.layers[i].w_grad and
            self.layers[i].b_grad, ready to be extracted using the
            get_gradient_vector method
        """
        # Perform forward propagation to calculate activations
        y = self.forward_prop(x)
        # Calculate the output layer delta and gradients
        self.layers[-1].delta = np.multiply(
            self.error_func.dEdy(y, target),
            self.layers[-1].act_func.dydx(self.layers[-1].pre_activation)
        )
        self.layers[-1].calc_gradients()
        # Calculate deltas and gradients for hidden layers
        for i in reversed(range(self.num_layers - 1)):
            self.layers[i].backprop(self.layers[i + 1])
            self.layers[i].calc_gradients()
        
        return y

    def get_parameter_vector(self):
        """
        get_parameter_vector: return all the parameters in the network as a long
        1D vector

        TODO: Alternative to nested list comprehension? Attribute which is
        initialised along with the network which is a view into the layer
        objects that automatically updates when the weights are modified? It's
        plausible that block initialises new memory every time it is called

        Inputs: None

        Outputs:
        -   parameter_vector: numpy array with shape (num_params, ) containing
            all of the parameters in the network, in the order: layer 0 weights,
            layer 0 bias, layer 1 weights, layer 1 bias, ..., final layer
            weights, final layer bias
        """
        return np.block([
            params for layer in self.layers for params in (
                layer.weights.ravel(), layer.bias.ravel()
            )
        ])

    def get_gradient_vector(self, x, target):
        """
        get_gradient_vector: return the mean (across data points) of the
        gradients (of the error between the targets and the network's
        predictions based on x) with respect to all of the parameters in the
        network as a long 1D vector

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
        return np.block([
            grads for layer in self.layers for grads in (
                layer.w_grad.mean(axis=-1).ravel(),
                layer.b_grad.mean(axis=-1).ravel()
            )
        ])

    def get_hessian(self): raise NotImplementedError

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
        v_pointer = 0
        for layer in self.layers:
            # Calculate the number of weight and bias parameters in the layer
            n_w, n_b = layer.output_dim * layer.input_dim, layer.output_dim
            # Set the weights and update the pointer
            layer.weights = new_parameters[v_pointer:v_pointer+n_w].reshape(
                layer.output_dim, layer.input_dim
            )
            v_pointer += n_w
            # Set the biases and update the pointer
            layer.bias = new_parameters[v_pointer:v_pointer+n_b].reshape(
                layer.output_dim, 1
            )
            v_pointer += n_b
    
    def mean_error(self, x, t):
        """
        mean_error: for a given set of inputs, calculate the network predictions
        and the error between the network predictions and the targets, returning
        the mean of the error across all data points

        Inputs:
        -   x: input to the neural network. Should be a numpy array with shape
            (input_dim, N_D)
        -   target: targets that the neural network is trying to predict. Should
            be a numpy array with shape (output_dim, N_D)

        Outputs:
        -   e: mean error across all data points as a numpy float64
        """
        y = self.forward_prop(x)
        return self.error_func(y, t).mean()

    def save_model(self, filename, filename_prepend_timestamp=True):
        # Save params, self.num_units_list, and identities of activation
        # functions as np.ndarrays in an npz file, and return the filename
        # params = self.get_parameter_vector()
        raise NotImplementedError
    
    def load_model(self, filename):
        # params = np.load(filename)
        # self.set_parameter_vector(params)
        raise NotImplementedError

    def print_weights(self):
        for i, layer in enumerate(self.layers):
            print(
                "Layer {}/{}:".format(i + 1, self.num_layers), "Weights:",
                layer.weights, "Biases:", layer.bias, "-" * 50, sep="\n"
            )

    def print_grads(self):
        for i, layer in enumerate(self.layers):
            print(
                "Layer {}/{}:".format(i + 1, self.num_layers),
                "Weight gradients:", layer.w_grad, "Bias gradients:",
                layer.b_grad, "-" * 50, sep="\n"
            )
    
class Dinosaur():
    pass

if __name__ == "__main__":
    np.random.seed(0)
    input_dim, output_dim, N_D = 2, 3, 4
    n = NeuralNetwork(
        # num_hidden_units=[], input_dim=input_dim, output_dim=output_dim, act_funcs=[a.Logistic()]
        num_hidden_units=[3, 4, 5], input_dim=input_dim, output_dim=output_dim
    )
    n.print_weights()

    x = np.random.normal(size=[input_dim, N_D])
    t = np.random.normal(size=[output_dim, N_D])
    y = n(x)
    assert y.shape == (output_dim, N_D)
    n.back_prop(x, t)
    n.print_grads()
    
    print(x, y, t, "Mean error = {:.5f}".format(n.mean_error(x, t)), sep="\n")

    # # n.plot_evals("Data/evaluations")
    # n.eval_gradient(0.4, 1.3)
    # for m in [n.weights, n.biases, n.activations, n.layer_outputs, n.deltas, n.w_grad]:
    #     print([i.shape for i in m])
    # dataset = d.DataSet("Data/sinusoidal_data_set.npz")
    # for _ in range(20):
    #     n.gradient_descent(dataset, n_its=500)
    #     n.plot_evals("Data/trained evaluations", xlims=[-0.5, 1.5])
