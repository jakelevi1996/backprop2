"""
Train a neural network on 1D sinusoidal data, and plot the predictions.
"""
import os
import numpy as np
if __name__ == "__main__":
    import __init__
from models import NeuralNetwork
import activations, data, optimisers, plotting

# Set time limit for training and evaluation frequency
t_lim = 5
t_interval = t_lim / 10

# Get name of output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs")

# Set the random seed
np.random.seed(2865)

# Generate random network and data
n = NeuralNetwork(
    input_dim=1,
    output_dim=1,
    num_hidden_units=[10],
    act_funcs=[activations.Cauchy(), activations.Identity()]
)
sin_data = data.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)

# Call gradient descent function
optimisers.gradient_descent(
    n,
    sin_data,
    terminator=optimisers.Terminator(t_lim=t_lim),
    evaluator=optimisers.Evaluator(t_interval=t_interval),
    result=optimisers.Result(name="SGD with line search", verbose=True),
    line_search=optimisers.LineSearch()
)

# Plot predictions
x_pred = np.linspace(-2, 2, 200).reshape(1, -1)
y_pred = n.forward_prop(x_pred)
plotting.plot_1D_regression(
    "Gradient descent predictions for 1D sin data",
    output_dir,
    sin_data,
    x_pred,
    y_pred
)
