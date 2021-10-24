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
