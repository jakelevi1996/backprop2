import numpy as np
import matplotlib.pyplot as plt

import activations as a

class NeuralNetwork():
    def __init__(
        self, num_hidden_units=[10], output_dim=1, input_dim=1,
        weight_std=1.0, bias_std=1.0,
        hidden_act=a.Logistic(), output_act=a.Identity(),
    ):
        # self.nhu = num_hidden_units
        # self.input_dim = input_dim
        # self.output_dim = output_dim
        # self.weight_std = weight_std
        # self.bias_std = bias_std
        self.initialise_weights(
            num_hidden_units, weight_std, bias_std, input_dim, output_dim
        )

    def initialise_weights(
        self, nhu, weight_std, bias_std, input_dim, output_dim
    ):
        self.weights = [np.hstack([
            np.random.normal(scale=weight_std, size=[nhu[0], input_dim]),
            np.random.normal(scale=bias_std, size=[nhu[0], 1])
        ])]
        for i in range(1, len(nhu)):
            self.weights.append(np.hstack([
                np.random.normal(scale=weight_std, size=[nhu[i], nhu[i - 1]]),
                np.random.normal(scale=bias_std, size=[nhu[i], 1])
            ]))
        self.weights.append(np.hstack([
            np.random.normal(scale=weight_std, size=[output_dim, nhu[-1]]),
            np.random.normal(scale=bias_std, size=[output_dim, 1])
        ]))
    
    def print_weights(self):
        print(len(self.weights))
        for w in self.weights: print(w)


if __name__ == "__main__":
    n = NeuralNetwork(num_hidden_units=[4, 4, 4], input_dim=3, output_dim=2)
    n.print_weights()