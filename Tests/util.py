import numpy as np
from models import NeuralNetwork

def get_random_network(low=3, high=6):
    """
    Generate a neural network with a random number of inputs, outputs, and
    hidden layers
    """
    input_dim = np.random.randint(low, high)
    output_dim = np.random.randint(low, high)
    num_hidden_layers = np.random.randint(low, high)
    num_hidden_units = np.random.randint(low, high, num_hidden_layers)
    n = NeuralNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        num_hidden_units=num_hidden_units
    )
    return n
