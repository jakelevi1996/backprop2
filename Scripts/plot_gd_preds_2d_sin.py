"""
Train a neural network on sinusoidal data with 2D inputs and 3D outputs, and
plot the predictions.
"""
import os
import numpy as np
if __name__ == "__main__":
    import __init__
from models import NeuralNetwork
import activations, data, optimisers, plotting

# Set time limit for training and evaluation frequency
t_lim = 10
t_interval = t_lim / 10

# Get name of output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs")

# Set the random seed
np.random.seed(2865)

# Generate random network and data
output_dim = 3
sin_data = data.SinusoidalDataSet2DnD(
    nx0=50,
    x0lim=[-2, 2],
    nx1=50,
    x1lim=[-2, 2],
    noise_std=0.1,
    train_ratio=0.8,
    output_dim=output_dim
)
n = NeuralNetwork(
    input_dim=2,
    output_dim=output_dim,
    num_hidden_units=[20, 20],
    act_funcs=[activations.Cauchy(), activations.Identity()]
)

# Call gradient descent function
result = optimisers.gradient_descent(
    n,
    sin_data,
    terminator=optimisers.Terminator(t_lim=t_lim),
    evaluator=optimisers.Evaluator(t_interval=t_interval),
    result=optimisers.Result(name="SGD with line search", verbose=True),
    line_search=optimisers.LineSearch()
)

# Plot predictions
x_pred = sin_data.x_test
y_pred = n.forward_prop(x_pred)
plotting.plot_2D_nD_regression(
    "Gradient descent predictions 2D-{}D sinusoid".format(output_dim),
    output_dir,
    n_output_dims=output_dim,
    dataset=sin_data,
    y_pred=y_pred
)
