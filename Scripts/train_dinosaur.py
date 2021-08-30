import numpy as np
if __name__ == "__main__":
    import __init__
import models
import data
import optimisers

# Define input and output dimensions for models and data
input_dim = 2
output_dim = 1

# Initialise models
num_hidden_units = [1]
network = models.NeuralNetwork(
    input_dim=input_dim,
    output_dim=output_dim,
    num_hidden_units=num_hidden_units,
)
dinosaur = models.Dinosaur(network)

# Initialise task set
task_set = data.TaskSet()
for x in [0.3, 0.9]:
    for y in [0.5, 0.7]:
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
        task_set.add_task(task)

for _ in range(10):
    # Perform one outer-loop iteration of meta-learning
    dinosaur.meta_learn(task_set, terminator=optimisers.Terminator(i_lim=1))
    # Check that the mean and scale are converging to sensible values
    print(dinosaur.mean)
    print(dinosaur.scale)
    # TODO: adapt to one test task that matches the distribution of training
    # tasks, and one test task that does not match that distribution, and
    # verify that a better reconstruction error, regularisation error, and
    # overall error is found for the test task that matches the distribution of
    # training tasks
