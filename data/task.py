""" Module containing the TaskSet class for meta-learning """

class TaskSet:
    def __init__(self):
        """ Initialise a TaskSet object """
        self.task_list = []
    
    def add_task(self, task):
        """ Add a task to this TaskSet object """
        self.task_list.append(task)
