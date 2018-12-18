import numpy as np
from numpy.random import normal as randn
import matplotlib.pyplot as plt

import activations as a

class NeuralNetwork():
    def __init__(
        self, num_hidden_units=[10], output_dim=1, input_dim=1,
        weight_std=1.0, bias_std=1.0,
        hidden_act=a.Logistic(), output_act=a.Identity(),
    ):
        # self.nhu = num_hidden_units
        self.input_dim = input_dim
        num_hidden_units = list(num_hidden_units)
        self.num_layers = len(num_hidden_units) + 1
        self.output_dim = output_dim
        # self.weight_std = weight_std
        # self.bias_std = bias_std
        self.initialise_weights(
            num_hidden_units, weight_std, bias_std, input_dim, output_dim
        )
        self.hidden_act = hidden_act
        self.output_act = output_act

    def initialise_weights(
        self, nhu, weight_std, bias_std, input_dim, output_dim
    ):
        # First layer weights and biases (from input):
        self.weights = [randn(0, weight_std, [nhu[0], input_dim])]
        self.biases = [randn(0, bias_std, [nhu[0], 1])]
        # Connections between hidden units:
        for i in range(1, len(nhu)):
            self.weights.append(randn(0, weight_std, [nhu[i], nhu[i - 1]]))
            self.biases.append(randn(0, bias_std, [nhu[i], 1]))
        # Final layer weights and biases (to output):
        self.weights.append(randn(0, weight_std, [output_dim, nhu[-1]]))
        self.biases.append(randn(0, bias_std, [output_dim, 1]))
    
    def print_weights(self):
        print(len(self.weights))
        for w, b in zip(self.weights, self.biases): print(w, "\n", b, "\n\n")
    
    def forward_prop(self, x):
        # Eventually this will be a wrapper for a forward-prop function used
        # for back-prop
        # assert type(x) is np.ndarray
        # if x.ndims > 1:
        #     assert x.ndims == 2
        #     assert x.shape[1] == self.input_dim
        
        # TODO: accept arrays of multiple inputs? Or will this not be useful
        # for forward-propagation?

        # Is it the linear combination of inputs we want to store, or that
        # value transformed by the activation function...?
            
        self.activations = [self.weights[0].dot(x) + self.biases[0]]

        for i in range(1, self.num_layers):
            layer_input = self.hidden_act(self.activations[-1])
            self.activations.append(
                self.weights[i].dot(layer_input) + self.biases[i]
            )

        return self.output_act(self.activations[-1]).reshape(self.output_dim)
    
    def plot_evals(self, filename, xlims=[-10, 10]):
        x = np.linspace(*xlims, 200)
        y = list(map(self.forward_prop, x))
        plt.figure(figsize=[8, 6])
        plt.plot(x, y, 'b')
        plt.title("Function evaluations")
        plt.grid(True)
        plt.savefig(filename)
        plt.close()


if __name__ == "__main__":
    n = NeuralNetwork(num_hidden_units=[4, 4, 4])
    # n.print_weights()

    n.plot_evals("Data/evluations")
