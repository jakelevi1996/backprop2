""" This module contains the Dinosaur class for meta-learning """
import optimisers

class Dinosaur:
    """ Dinosaur class for meta-learning """

    def __init__(
        self,
        network,
        regulariser,
        primary_initialisation_task,
        secondary_initialisation_task,
        batch_size=50,
        t_lim=None,
    ):
        """ Initialise a dinosaur object """
        self._network = network
        self._regulariser = regulariser
        self._batch_size = batch_size
        self._evaluator = optimisers.Evaluator(t_interval=0.1)
        self._result = optimisers.Result(name="Dinosaur")
        if t_lim is not None:
            self._timer = optimisers.Timer(t_lim)
            self._timer.begin()
        else:
            self._timer = None

        # Get parameters from optimising the first initialisation task
        line_search = optimisers.LineSearch()
        dynamic_terminator = self._get_terminator(
            primary_initialisation_task,
        )
        # self._result.add_column(optimisers.results.columns.BatchImprovementProbability(dynamic_terminator))
        optimisers.GradientDescent(line_search).optimise(
            model=network,
            dataset=primary_initialisation_task,
            evaluator=self._evaluator,
            terminator=dynamic_terminator,
            result=self._result,
            batch_getter=dynamic_terminator,
        )
        w1 = network.get_parameter_vector().copy()
        
        # Get parameters from optimising the second initialisation task
        dynamic_terminator = self._get_terminator(
            secondary_initialisation_task,
        )
        optimisers.GradientDescent(line_search).optimise(
            model=network,
            dataset=secondary_initialisation_task,
            evaluator=self._evaluator,
            terminator=dynamic_terminator,
            result=self._result,
            batch_getter=dynamic_terminator,
        )
        w2 = network.get_parameter_vector()

        # Set the regulariser parameters and add to the network
        self._regulariser.update([w1, w2])
        self._network.set_regulariser(self._regulariser)

    
    def meta_learn(self, task_set, terminator=None):
        """ Learn meta-parameters for a task-set """
    
    def fast_adapt(self, data_set):
        """ Adapt to a data set, given the current meta-parameters """

    def _get_terminator(self, dataset):
        if self._timer is not None:
            t_lim = self._timer.time_remaining()
        else:
            t_lim = None
        terminator = optimisers.DynamicTerminator(
            model=self._network,
            dataset=dataset,
            batch_size=self._batch_size,
            replace=True,
            t_lim=t_lim,
        )
        return terminator
