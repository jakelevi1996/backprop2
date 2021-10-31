from data.regression import Regression
from data.mnist import Mnist

class Three(Regression):
    """ Regression dataset consisting of an image of a handwritten number 3,
    taken from the MNIST dataset. The exact image that is used will vary
    depending on the status of the numpy random seed. """
    def __init__(self, **_):
        Regression.__init__(self, input_dim=2, output_dim=1)
        mnist_task_map = Mnist()
        three_dataset = mnist_task_map.train.get_batch(
            label=3,
            batch_size=1,
        )[0]
        self.train = three_dataset.train
        self.test  = three_dataset.test
