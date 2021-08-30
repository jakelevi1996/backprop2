""" This module contains the Dinosaur class for meta-learning """

class Dinosaur:
    """ Dinosaur class for meta-learning """

    def __init__(self, network):
        """ Initialise a dinosaur object """
        self._network = network
        self.mean = None
        self.scale = None
    
    def meta_learn(self, task_set, terminator=None):
        """ Learn meta-parameters for a task-set """
    
    def fast_adapt(self, data_set):
        """ Adapt to a data set, given the current meta-parameters """
