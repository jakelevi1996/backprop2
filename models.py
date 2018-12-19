import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal as randn

import activations as a
import errors as e
import data as d

class NeuralNetwork():
    def __init__(
        self, num_hidden_units=[10], output_dim=1, input_dim=1,
        weight_std=1.0, bias_std=1.0,
        hidden_act=a.Logistic(), output_act=a.Identity(),
        error_func=e.SumOfSquares()
    ):
        num_hidden_units = list(num_hidden_units)
        self.num_layers = len(num_hidden_units) + 1
        # Set input and output dimensions:
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Randomly initialise weights:
        self.initialise_weights(num_hidden_units, weight_std, bias_std)
        # Set activation and error functions:
        self.hidden_act = hidden_act
        self.output_act = output_act
        self.error_func = error_func

    def initialise_weights(self, nhu, weight_std, bias_std):
        # First layer weights and biases (from input):
        self.weights = [randn(0, weight_std, [nhu[0], self.input_dim])]
        self.biases = [randn(0, bias_std, [nhu[0], 1])]
        # Connections between hidden units:
        for i in range(1, len(nhu)):
            self.weights.append(randn(0, weight_std, [nhu[i], nhu[i - 1]]))
            self.biases.append(randn(0, bias_std, [nhu[i], 1]))
        # Final layer weights and biases (to output):
        self.weights.append(randn(0, weight_std, [self.output_dim, nhu[-1]]))
        self.biases.append(randn(0, bias_std, [self.output_dim, 1]))
    
    def forward_prop(self, x):
        # Calculate activations and output from input layer
        self.layer_outputs = []
        self.activations = [self.weights[0].dot(x) + self.biases[0]]
        # Calculate activations and output from hidden layers
        for w, b in zip(self.weights[1:], self.biases[1:]):
            self.layer_outputs.append(self.hidden_act(self.activations[-1]))
            self.activations.append(w.dot(self.layer_outputs[-1]) + b)
        # Calculate activations and output from output layer
        self.layer_outputs.append(self.output_act(self.activations[-1]))

        return self.layer_outputs[-1].reshape(self.output_dim)
    
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


    def gradient_descent(
        self, x_list, t_list, n_its=2000, learning_rate=5e-1, print_every=100
    ):
        for i in range(n_its):
            if i % print_every == 0: print("Iteration", i)
            # Calculate gradients for first data point:
            self.eval_gradient(x_list[0], t_list[0])
            deltas = self.deltas
            w_grads = self.w_grad
            # Accumulate gradients for remaining data points:
            for x, t in zip(x_list[1:], t_list[1:]):
                self.eval_gradient(x, t)
                for i, (d, wg) in enumerate(zip(self.deltas, self.w_grad)):
                    deltas[i] += d
                    w_grads[i] += wg
            # Descend gradient:
            for i, (d, wg) in enumerate(zip(deltas, w_grads)):
                self.weights[i] -= learning_rate * wg / len(x_list)
                self.biases[i] -= learning_rate * d / len(x_list)

    def linesearch_condition(self, ): pass

    def gradient_descent_backtracking(self, x, t, ): pass
    
    def save_model(self, filename): pass
    
    def load_model(self, filename): pass

    
    def plot_evals(self, filename, xlims=[-10, 10]):
        x = np.linspace(*xlims, 200)
        y = list(map(self.forward_prop, x))
        plt.figure(figsize=[8, 6])
        plt.plot(x, y, 'b')
        plt.title("Function evaluations")
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    def print_weights(self):
        print(len(self.weights))
        for w, b in zip(self.weights, self.biases): print(w, "\n", b, "\n\n")
    



if __name__ == "__main__":
    np.random.seed(0)
    n = NeuralNetwork(num_hidden_units=[10])
    # n.print_weights()

    # n.plot_evals("Data/evaluations")
    n.eval_gradient(0.4, 1.3)
    for m in [n.weights, n.biases, n.activations, n.layer_outputs, n.deltas, n.w_grad]:
        print([i.shape for i in m])
    
    s = d.DataSet("Data/sinusoidal_data_set.npz")
    for _ in range(10):
        n.gradient_descent(s.x_train, s.y_train, n_its=500)
        n.plot_evals("Data/trained evaluations", xlims=[-0.5, 1.5])

