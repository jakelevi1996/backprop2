import os
import shutil
import numpy as np
if __name__ == "__main__":
    import __init__
import models
import data
import optimisers
import plotting

np.random.seed(0)

# Define input and output dimensions for models and data
input_dim = 2
output_dim = 1

# Initialise network model
num_hidden_units = [10]
network = models.NeuralNetwork(
    input_dim=input_dim,
    output_dim=output_dim,
    num_hidden_units=num_hidden_units,
)

# Get directory for saving outputs to disk
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(
    current_dir,
    "Outputs",
    "Train Dinosaur",
)
if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)
print("Saving output plots in \"%s\"" % output_dir)
os.system("explorer \"%s\"" % output_dir)

# Initialise task set
task_set = data.TaskSet()
for x in [0.3, 0.9]:
    for y in [0.5, 0.7]:
        # Initialise task
        task = data.GaussianCurve(
            input_dim=input_dim,
            output_dim=output_dim,
            n_train=2000,
            n_test=2000,
            x_lo=-1,
            x_hi=1,
            input_offset=np.array([x, y]).reshape([input_dim, 1]),
            input_scale=5,
            output_offset=0,
            output_scale=10,
        )
        # Add task to task set
        task_set.add_task(task)
        # Plot task for reference and save to disk
        plotting.plot_2D_regression(
            "Dinosaur task, x = %.1f, y = %.1f" % (x, y),
            output_dir,
            task,
            output_dim,
            model=network,
        )

# Initialise meta-learning model
regulariser = models.dinosaur.regularisers.Quadratic()
dinosaur = models.Dinosaur(
    network=network,
    regulariser=regulariser,
    primary_initialisation_task=task_set.task_list[0],
    secondary_initialisation_task=task_set.task_list[1],
    t_lim=10,
)

for _ in range(10):
    # Perform one outer-loop iteration of meta-learning
    dinosaur.meta_learn(task_set, terminator=optimisers.Terminator(i_lim=1))
    # Check that the mean and scale are converging to sensible values
    print(regulariser.mean)
    print(regulariser.scale)
    # TODO: adapt to one test task that matches the distribution of training
    # tasks, and one test task that does not match that distribution, and
    # verify that a better reconstruction error, regularisation error, and
    # overall error is found for the test task that matches the distribution of
    # training tasks


# Plot task predictions after meta-learning
for i, task in enumerate(task_set.task_list):
    dinosaur.fast_adapt(task)
    plotting.plot_2D_regression(
        "Dinosaur task %i" % i,
        output_dir,
        task,
        output_dim,
        model=network,
    )
