"""
Train a neural network on sinusoidal data with 2D inputs and 3D outputs, and
plot the DBS metric vs iteration number.
"""
import os
import numpy as np
if __name__ == "__main__":
    import __init__
import models, data, optimisers, plotting

# Set time limit for training and evaluation frequency
i_lim = 50000
i_interval = i_lim / 100

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

# Create result object and add columns for iteration and DBS
result = optimisers.Result(
    name="SGD with line search",
    verbose=True,
    add_default_columns=False
)
i_column    = optimisers.results.columns.Iteration()
dbs_column  = optimisers.results.columns.DbsMetric()
result.add_column(i_column)
result.add_column(dbs_column)

# Call gradient descent function
model.get_gradient_vector(sin_data.x_train, sin_data.y_train)
result = optimisers.gradient_descent(
    model,
    sin_data,
    terminator=optimisers.Terminator(i_lim=i_lim),
    evaluator=optimisers.Evaluator(i_interval=i_interval),
    result=result,
    line_search=optimisers.LineSearch(),
    batch_getter=optimisers.batch.ConstantBatchSize(50)
)

# Plot DBS metric vs iteration
plotting.plot_result_attribute(
    "Gradient descent DBS metric for 2D-%iD sinusoid" % output_dim,
    output_dir,
    [result],
    dbs_column.name
)
