"""
Train a neural network on sinusoidal data with 2D inputs and 3D outputs, and
plot the predictions.
"""
import os
import numpy as np
if __name__ == "__main__":
    import __init__
import models, data, optimisers, plotting

# Set time limit for training and evaluation frequency
t_lim = 10
t_interval = t_lim / 10

# Get name of output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs")

# Set the random seed
np.random.seed(2865)

# Generate random network and data
input_dim = 2
output_dim = 3
x_lo = -2
x_hi = 2
sin_data = data.Sinusoidal(
    input_dim=input_dim,
    output_dim=output_dim,
    x_lo=x_lo,
    x_hi=x_hi
)
model = models.NeuralNetwork(
    input_dim=input_dim,
    output_dim=output_dim,
    num_hidden_units=[20, 20],
    act_funcs=[models.activations.cauchy, models.activations.identity]
)

# Call gradient descent function
result = optimisers.gradient_descent(
    model,
    sin_data,
    terminator=optimisers.Terminator(t_lim=t_lim),
    evaluator=optimisers.Evaluator(t_interval=t_interval),
    result=optimisers.Result(name="SGD with line search", verbose=True),
    line_search=optimisers.LineSearch(),
    batch_getter=optimisers.batch.ConstantBatchSize(50)
)

# Plot predictions
x_pred_0 = x_pred_1 = np.linspace(x_lo, x_hi)
plotting.plot_2D_nD_regression(
    "Gradient descent predictions for 2D-%iD sinusoid" % output_dim,
    output_dir,
    n_output_dims=output_dim,
    dataset=sin_data,
    x_pred_0=x_pred_0,
    x_pred_1=x_pred_1,
    model=model
)

# Plot learning curve
plotting.plot_training_curves(
    [result],
    "Gradient descent learning curves for 2D-%iD sinusoid" % output_dim,
    output_dir,
    e_lims=[0, 0.5]
)
