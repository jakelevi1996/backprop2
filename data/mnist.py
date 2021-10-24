import os
import pickle
import numpy as np
from data.task import TaskMap, TaskSubMap
from data.dataset import DataSet

CURRENT_DIR         = os.path.dirname(os.path.abspath(__file__))
MNIST_DATA_DIRNAME  = os.path.join(CURRENT_DIR, "saved_data")
MNIST_DATA_FILENAME = os.path.join(MNIST_DATA_DIRNAME, "mnist.pkl")

MNIST_N_TRAIN       = 60000
MNIST_N_TEST        = 10000
MNIST_WIDTH_PIXELS  = 28
MNIST_HEIGHT_PIXELS = 28
MNIST_INPUT_DIM     = 2
MNIST_OUTPUT_DIM    = 1

class Mnist(TaskMap):
    def __init__(self, num_validation=MNIST_N_TEST):
        if os.path.exists(MNIST_DATA_FILENAME):
            with open(MNIST_DATA_FILENAME, "rb") as f:
                (
                    self.task_map_train,
                    self.task_map_test,
                    self.task_map_validation,
                ) = pickle.load(f)


            assert (
                self.task_map_train.num_tasks
                + self.task_map_validation.num_tasks
                == MNIST_N_TRAIN
            )
            assert self.task_map_test.num_tasks == MNIST_N_TEST

            if self.task_map_validation.num_tasks != num_validation:
                self._gen_data(num_validation)

        else:
            self._gen_data(num_validation)

    def _gen_data(self, num_validation):
        import tensorflow as tf

        mnist_data = tf.keras.datasets.mnist.load_data()
        (y_train, train_labels), (y_test, test_labels) = mnist_data
        assert y_train.shape == (
            MNIST_N_TRAIN,
            MNIST_HEIGHT_PIXELS,
            MNIST_WIDTH_PIXELS,
        )
        assert y_test.shape == (
            MNIST_N_TEST,
            MNIST_HEIGHT_PIXELS,
            MNIST_WIDTH_PIXELS,
        )
        n = MNIST_WIDTH_PIXELS * MNIST_HEIGHT_PIXELS
        y_train = y_train.reshape(MNIST_N_TRAIN, MNIST_OUTPUT_DIM, n) / 255.0
        y_test  =  y_test.reshape(MNIST_N_TEST , MNIST_OUTPUT_DIM, n) / 255.0

        validation_inds = np.random.choice(
            MNIST_N_TRAIN,
            num_validation,
            replace=False,
        )
        validation_bool_inds = np.full(MNIST_N_TRAIN, False)
        validation_bool_inds[validation_inds] = True
        y_validation        = y_train[validation_bool_inds]
        y_train             = y_train[~validation_bool_inds]
        validation_labels   = train_labels[validation_bool_inds]
        train_labels        = train_labels[~validation_bool_inds]

        x0 = np.linspace(-1, 1, MNIST_WIDTH_PIXELS)
        x1 = np.linspace(-1, 1, MNIST_HEIGHT_PIXELS)
        xx = np.meshgrid(x0, x1)
        x = np.stack([xx[0].ravel(), np.flip(xx[1].ravel())], axis=0)
        assert x.shape == (MNIST_INPUT_DIM, n)

        self.task_map_train = TaskSubMap()
        self.task_map_test = TaskSubMap()
        self.task_map_validation = TaskSubMap()

        for task_submap, y_subset, label_subset in [
            [self.task_map_train,       y_train,        train_labels        ],
            [self.task_map_test,        y_test,         test_labels         ],
            [self.task_map_validation,  y_validation,   validation_labels   ],
        ]:
            for y, label in zip(y_subset, label_subset):
                task = DataSet()
                task.input_dim  = MNIST_INPUT_DIM
                task.output_dim = MNIST_OUTPUT_DIM
                for data_subset in [task.train, task.test]:
                    data_subset.x = x
                    data_subset.y = y
                    data_subset.n = n
                    data_subset.label = label
                task_submap.add_task(task, label)

        if not os.path.isdir(MNIST_DATA_DIRNAME):
            os.makedirs(MNIST_DATA_DIRNAME)

        with open(MNIST_DATA_FILENAME, "wb") as f:
            pickle.dump(
                [
                    self.task_map_train,
                    self.task_map_test,
                    self.task_map_validation,
                ],
                f,
            )
