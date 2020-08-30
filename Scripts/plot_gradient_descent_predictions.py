import os
import numpy as np
if __name__ == "__main__":
    import __init__
from models import NeuralNetwork
import activations, data, optimisers, plotting

# Get name of output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs")

# Set the random seed
seed = 2865
np.random.seed(seed)
# Generate random network, data, and number of iterations
n = NeuralNetwork(
    1, 1, [10], [activations.Gaussian(), activations.Identity()]
)
sin_data = data.SinusoidalDataSet1D1D(xlim=[-2, 2], freq=1)
n_iters = 500
# Call gradient descent function
result_ls = optimisers.gradient_descent(
    n,
    sin_data,
    n_iters=n_iters,
    eval_every=10,
    verbose=True,
    name="SGD with line search",
    line_search_flag=True
)
# Plot predictions
x_pred = np.linspace(-2, 2).reshape(1, -1)
y_pred = n.forward_prop(x_pred)
plotting.plot_1D_regression(
    "Gradient descent predictions for sinusoidal",
    output_dir,
    sin_data,
    x_pred,
    y_pred
)
