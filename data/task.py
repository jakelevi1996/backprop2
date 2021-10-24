""" Module containing the TaskSet class for meta-learning, and the TaskMap and
TaskSubMap classes for dynamic memory classification """
import numpy as np

class TaskSet:
    def __init__(self):
        """ Initialise a TaskSet object """
        self.task_list = []

    def add_task(self, task):
        """ Add a task to this TaskSet object """
        self.task_list.append(task)

    def get_batch(self, batch_size, replace=False):
        batch_inds = np.random.choice(
            len(self.task_list),
            batch_size,
            replace,
        )
        task_batch = [self.task_list[i] for i in batch_inds]
        return task_batch

class TaskSubMap:
    def __init__(self):
        self.dict = dict()
        self.num_tasks = 0

    def add_task(self, task, label):
        if label in self.dict:
            self.dict[label].add_task(task)
        else:
            self.dict[label] = TaskSet()
            self.dict[label].add_task(task)

        self.num_tasks += 1

class TaskMap:
    def __init__(self):
        self.train        = TaskSubMap()
        self.test         = TaskSubMap()
        self.validation   = TaskSubMap()
