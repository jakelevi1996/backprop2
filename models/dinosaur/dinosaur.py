""" This module contains the Dinosaur class for meta-learning """
from optimisers import gradient_descent

class Dinosaur:
    """ Dinosaur class for meta-learning """

    def __init__(
        self,
        network,
        regulariser,
        primary_initialisation_task,
        secondary_initialisation_task,
    ):
        """ Initialise a dinosaur object """
        self._network = network
        self._regulariser = regulariser
    
    def meta_learn(self, task_set, terminator=None):
        """ Learn meta-parameters for a task-set """
    
    def fast_adapt(self, data_set):
        """ Adapt to a data set, given the current meta-parameters """
