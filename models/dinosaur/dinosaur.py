""" This module contains the Dinosaur class for meta-learning """
import optimisers
from models.dinosaur.regularisers import Eve

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
        self._initialised_regulariser = False
        self._line_search = optimisers.LineSearch()
        self._result.add_column(
            optimisers.results.columns.StepSize(self._line_search)
        )
        self._optimiser = optimisers.GradientDescent(self._line_search)
        if t_lim is not None:
            self._timer = optimisers.Timer(t_lim)
            self._timer.begin()
        else:
            self._timer = None

        self._terminator = optimisers.DynamicTerminator(
            model=self._network,
            dataset=primary_initialisation_task,
            batch_size=self._batch_size,
            replace=True,
            t_lim=t_lim,
        )
        self._result.add_column(
            optimisers.results.columns.BatchImprovementProbability(
                self._terminator
            )
        )

        # Get parameters from optimising the first initialisation task
        self._reset_terminator = False
        self.fast_adapt(primary_initialisation_task)
        self._reset_terminator = True
        w1 = network.get_parameter_vector().copy()

        # Get parameters from optimising the second initialisation task
        dE = self.fast_adapt(secondary_initialisation_task)
        w2 = network.get_parameter_vector()

        # Set the regulariser parameters and add to the network
        self._regulariser.update([w1, w2], [dE])
        if isinstance(self._regulariser, Eve):
            self._optimiser = self._regulariser
            self._optimiser.set_line_search(self._line_search)
            eve_column = optimisers.columns.EveConvergence(self._regulariser)
            self._result.add_column(eve_column)
        else:
            self._network.set_regulariser(self._regulariser)
            self._result.add_column(optimisers.columns.RegularisationError())
        self._initialised_regulariser = True


    def meta_learn(self, task_set, terminator=None):
        """ Learn meta-parameters for a task-set """
        # Initialise loop variables
        i = 0
        terminator.set_initial_iteration(i)
        # Start outer loop
        while True:
            # Initialise results lists for this iteration
            w_list = []
            dE_list = []
            # Iterate through tasks in the task set
            for task in task_set.task_list:
                # Check if ready to finish meta-learning
                if terminator.ready_to_terminate(i):
                    return

                # Adapt to the current task and store the results
                dE = self.fast_adapt(task)
                dE_list.append(dE)
                w = self._network.get_parameter_vector().copy()
                w_list.append(w)

            # Update meta-parameters and increment loop counter
            self._regulariser.update(w_list, dE_list)
            i += 1

    def fast_adapt(self, data_set):
        """ Adapt to a data set, given the current meta-parameters. If the
        regulariser for this Dinosaur object has already been initialised, then
        before adaptation, the parameters of this Dinosaur object's internal
        NeuralNetwork object are reset to the mean parameters according to the
        regulariser. This method returns the reduction in the mean
        reconstruction error which results from optimisation """
        # Optionally reset network parameters to the mean
        if self._initialised_regulariser:
            self._network.set_parameter_vector(self._regulariser.mean)

        # Initialise line search and dynamic terminator
        self._line_search.reset()
        if self._reset_terminator:
            if self._timer is not None:
                t_lim = self._timer.time_remaining()
            else:
                t_lim = None
            self._terminator.reset(data_set=data_set, t_lim=t_lim)

        # Optimise the model for the data set using gradient descent
        self._optimiser.optimise(
            model=self._network,
            dataset=data_set,
            evaluator=self._evaluator,
            terminator=self._terminator,
            result=self._result,
            batch_getter=self._terminator,
        )

        # Return the reduction in the mean reconstruction error
        mean_reconstruction_error_reduction = (
            self._terminator.initial_mean_reconstruction_error
            - self._result.get_final_train_reconstruction_error()
        )
        return mean_reconstruction_error_reduction
