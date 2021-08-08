if __name__ == "__main__":
    import __init__
import models
import data
import optimisers

dinosaur = models.Dinosaur()
task_set = data.TaskSet()
for x in [1, 3]:
    for y in [1.5, 2.5]:
        task_set.add_task(data.GaussianCurve(input_loc=[x, y]))

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
