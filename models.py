"""
Module to contain neural network classes, which themselves contain methods for
initialisation, evaluation, and gradient calculations.

Notation convention: it is assumed that each neural network has L hidden layers
and 1 output layer, with N_0 units in the first (input) hidden layer, N_1 units
in the second hidden layer, ..., N_L-1 units in the last hidden layer, and N_L
units in the output layer. Each model can accept multidimensional inputs and
outputs; the dimension of the input data is D_in, and the dimension of the
output data is N_L (== the number of units in the output layer). Each neural
network can accept multiple input points simultaneously; when inputs are fed to
a neural network, it is assumed that N_D input points are fed simultaneously in
a numpy.ndarray with shape [D_in, N_D] (IE the first array dimension refers to
the dimension within each input point, and the second array dimension refers to
a particular input point)*. When such an input is fed into the neural network,
the output will be a numpy.ndarray with shape [N_L, N_D]

*This convention is chosen to simplify matrix multiplication during
forward-propagation and back-propagation; it is possible that in future the
array shape dimensions will be swapped, and the matrix multiplication operations
will be carried out with np.einsum operations with subscripts [0, 1] and [2, 1]
(whereas conventional matrix multiplication is equivalent to np.einsum with
subscripts [0, 1] and [1, 2])
"""

import numpy as np
import matplotlib.pyplot as plt

import activations as a
import errors as e
import data as d

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
        -   input: input to the layer. Should be a numpy.ndarray with shape
            [self.input_dim, N_D]

        Outputs:
        -   self.output: output from the layer, in a numpy.ndarray with shape
            [self.output_dim, N_D]
        -   self.pre_activation (not returned): result of linear transformation
            being applied to the input, before nonlinearity is applied to give
            the output, in a numpy.ndarray with shape [self.output_dim, N_D]
        """
        self.input = input
        self.pre_activation = np.matmul(self.weights, input) + self.bias
        self.output = self.act_func(self.pre_activation)
        return self.output
    
    def backprop(self, next_layer):
        # Calculate deltas (gradient of error function WRT pre-activations) and
        # calculate gradientsof error function WRT current layer weights and
        # biases
        # TODO: does this work with multi-D inputs?
        self.deltas = self.act_func.dydx(self.pre_activation) * (
            next_layer.weights.T.dot(next_layer.deltas)
        )
        self.b_grad = self.deltas
        self.w_grad = np.outer(self.deltas, self.input)


class NeuralNetwork():
    def __init__(
        self, input_dim=1, output_dim=1, num_hidden_units=[10],
        weight_std=1.0, bias_std=1.0,
        # hidden_act=a.Logistic(), output_act=a.Identity(),
        act_funcs=[a.Logistic(), a.Identity()],
        error_func=e.SumOfSquares()
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

        # Initialise network layers
        act_funcs = resize_list(act_funcs, self.num_layers)
        self.layers = [NeuralLayer(
            self.num_units_list[0], input_dim, act_funcs[0],
            weight_std, bias_std
        )]
        for (num_units, num_inputs, act_func) in zip(
            self.num_units_list[1:], self.num_units_list[:-1], act_funcs[1:]
        ):
            self.layers.append(NeuralLayer(
                num_units, num_inputs, act_func, weight_std, bias_std
            ))

        # TODO: add options for different initialisers, including depending on
        # previous layers (see Glorot)

    def forward_prop(self, x):
        """
        forward_prop: propogate an input forward through the network, and return
        the output from the network.

        Inputs:
        -   x: input to the neural network. Should be a numpy.ndarray with shape
            [self.input_dim, N_D]

        Outputs:
        -   layer_output: output from the final layer in the network, == the
            network output. Should be in a numpy.ndarray with shape
            [self.output_dim, N_D]
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
        # Perform forward propagation to calculate unit activations:
        y = self.forward_prop(x)
        # Calculate delta for final layer (output) units:
        act_grad = self.output_act.dydx(self.activations[-1])
        deltas = [act_grad * self.error_func.dEdy(y, target)]
        # Calculate deltas for hidden units (NB list order is reversed):
        for i in range(self.num_layers - 1, 0, -1):
            act_grad = self.hidden_act.dydx(self.activations[i - 1])
            deltas.append(act_grad * self.weights[i].T.dot(deltas[-1]))
        self.deltas = list(reversed(deltas))
    
    def eval_gradient(self, x, target):
        self.back_prop(x, target)
        # Calculate input layer gradients:
        self.w_grad = [np.outer(self.deltas[0], x)]
        # Calculate all other gradients:
        for d, z, in zip(self.deltas[1:], self.layer_outputs[:-1]):
            self.w_grad.append(np.outer(d, z))

    def accumulate_gradients(self, x_list, t_list):
        # Calculate gradients for first data point:
        self.eval_gradient(x_list[0], t_list[0])
        w_grads = self.w_grad
        b_grads = self.deltas
        # Accumulate gradients for remaining data points:
        for x, t in zip(x_list[1:], t_list[1:]):
            self.eval_gradient(x, t)
            for i, (w, b) in enumerate(zip(self.w_grad, self.deltas)):
                w_grads[i] += w
                b_grads[i] += b
        
        return w_grads, b_grads
    
    def eval_mean_error(self, x_list, t_list):
        n = check_reduce(len(x_list), len(t_list))
        error = 0
        for x, t in zip(x_list, t_list):
            y = self.forward_prop(x)
            error += self.error_func(y, t)
        
        return error / n

    def get_parameter_vector(self):
        pass

    def get_gradient_vector(self):
        pass

    def get_hessian(self):
        pass

    def set_parameter_vector(self, new_parameters):
        pass
    
    # TODO: optimisation algorithms in optimisers module
    def gradient_descent(
        self, dataset, n_its=2000, learning_rate=5e-1, print_every=100,
        return_errors=False
    ):
        n = check_reduce(len(dataset.x_train), len(dataset.y_train))
        if return_errors:
            train_error = np.empty(n_its)
            test_error = np.empty(n_its)

        for i in range(n_its):
            if i % print_every == 0: print("Iteration", i)
            # Calculate gradients:
            w_grads, b_grads = self.accumulate_gradients(
                dataset.x_train, dataset.y_train
            )
            # Descend gradient:
            for j, (b, w) in enumerate(zip(b_grads, w_grads)):
                self.weights[j] -= learning_rate * w / n
                self.biases[j] -= learning_rate * b / n
            # Store mean errors for training and test sets:
            if return_errors:
                train_error[i] = self.eval_mean_error(
                    dataset.x_train, dataset.y_train
                )
                test_error[i] = self.eval_mean_error(
                    dataset.x_test, dataset.y_test
                )

        if return_errors: return train_error, test_error
    
    
    def save_model(self, filename, filename_prepend_timestamp=True):
        # params = self.get_parameter_vector()
        # Save params, self.num_units_list, and identities of activation
        # functions as np.ndarrays in an npz file
        pass
        return filename
    
    def load_model(self, filename):
        pass
        params = np.load(filename)
        self.set_parameter_vector(params)

    
    def plot_evals(self, filename, xlims=[-10, 10]):
        # TODO: move this to plotting module
        x = np.linspace(*xlims, 200)
        y = list(map(self.forward_prop, x))
        plt.figure(figsize=[8, 6])
        plt.plot(x, y, 'b')
        plt.title("Function evaluations")
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    def print_weights(self):
        for i, layer in enumerate(self.layers):
            print(
                "Layer {}/{}:".format(i + 1, self.num_layers), "Weights:",
                layer.weights, "Biases:", layer.bias, "-" * 50, sep="\n"
            )
    
class Dinosaur():
    pass

if __name__ == "__main__":
    np.random.seed(0)
    D_in, N_L, N_D = 2, 3, 7
    n = NeuralNetwork(
        num_hidden_units=[3, 4, 5], input_dim=D_in, output_dim=N_L
    )
    n.print_weights()

    x = np.random.normal(size=[D_in, N_D])
    y = n(x)
    print(x, y, y.shape, [N_L, N_D], sep="\n")

    # # n.plot_evals("Data/evaluations")
    # n.eval_gradient(0.4, 1.3)
    # for m in [n.weights, n.biases, n.activations, n.layer_outputs, n.deltas, n.w_grad]:
    #     print([i.shape for i in m])
    # dataset = d.DataSet("Data/sinusoidal_data_set.npz")
    # for _ in range(20):
    #     n.gradient_descent(dataset, n_its=500)
    #     n.plot_evals("Data/trained evaluations", xlims=[-0.5, 1.5])
