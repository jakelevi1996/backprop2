"""
Script to compare learning curves for different batch sizes, when training a
neural network on sinusoidal data with 2 dimensional inputs and 3 dimensional
outputs, using gradient descent with a line-search.

It is worth noting that the capacity of the neural network being used (IE how
many hidden units and hidden layers) seems to have quite a dramatic effect on
the output from this script. EG with only 1 hidden layer with 10 hidden units,
the batch size seems to have little effect on the final performance (in this
case the initial parameters seem to be a much more significant factor), whereas
for 2 hidden layers with 20 hidden units each, the relative effect of the batch
size is much more dramatic.

TODO: to what extent is the variation between repeats of experiments with the
same batch down to good/bad parameter initialisations? Try with a constant
pre-activation statistics initialisation, instead of constant parameter
statistics initialisation
"""
import os
import numpy as np
from time import perf_counter
if __name__ == "__main__":
    import __init__
from models import NeuralNetwork
import activations, data, optimisers, plotting

t_0 = perf_counter()

# Perform warmup experiment so process acquires priority
optimisers.warmup()

# Initialise data, time limit, and results list
np.random.seed(9251)
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
t_lim = 25
t_interval = t_lim / 50
results_list = []

# Initialise list of batch-getters
batch_size_list = np.linspace(50, sin_data.n_train, 5, endpoint=False)

for seed in [2295, 6997, 7681]:
    # Set the random seed
    np.random.seed(seed)
    # Generate random network and store initial parameters
    n = NeuralNetwork(
        input_dim=2,
        output_dim=output_dim,
        num_hidden_units=[20, 20],
        act_funcs=[activations.Cauchy(), activations.Identity()]
    )
    w0 = n.get_parameter_vector().copy()
    # Iterate through constant size batch-getters
    for batch_size in batch_size_list:
        # Set name for experiment
        name = "Batch size = {:04d}".format(int(batch_size))
        # Reset parameter vector
        n.set_parameter_vector(w0)
        # Call gradient descent function
        result = optimisers.gradient_descent(
            n,
            sin_data,
            terminator=optimisers.Terminator(t_lim=t_lim),
            evaluator=optimisers.Evaluator(t_interval=t_interval),
            result=optimisers.Result(name=name, verbose=True),
            line_search=optimisers.LineSearch(),
            batch_getter=optimisers.batch.ConstantBatchSize(int(batch_size))
        )
        results_list.append(result)
    
    # Try again with full training set
    n.set_parameter_vector(w0)
    result = optimisers.gradient_descent(
        n,
        sin_data,
        terminator=optimisers.Terminator(t_lim=t_lim),
        evaluator=optimisers.Evaluator(t_interval=t_interval),
        line_search=optimisers.LineSearch(),
        batch_getter=optimisers.batch.FullTrainingSet(),
        result=optimisers.Result(name="Full training set", verbose=True),
    )
    results_list.append(result)

# Get name of output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs")

# Compare training curves
plotting.plot_training_curves(
    results_list,
    "Comparing batch sizes for gradient descent on 2D sinusoidal data",
    output_dir,
    e_lims=[0, 4],
    tp=0.5
)

print("Script run in {:.3f} s".format(perf_counter() - t_0))
