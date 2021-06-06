from optimisers.timer import TimedObject

class AbstractResult(TimedObject):
    """ The AbstractResult is used as an abstract parent class for
    _optimiser.results.Result, and is also used as a blank result by
    optimisers.columns.OptimalBatchSize. This is to avoid the circular
    dependency of _optimiser.results needing to import columns so that it can
    have default columns, and optimisers.columns needing to import results so
    that optimisers.columns.OptimalBatchSize can use an empty result, which is
    more efficient than initiating a new result (and new column objects etc)
    every time the optimisation function is called """
    def __init__(self):
        """ Initialise an AbstractResult object, which by default is not
        verbose, and has already "begun" (so that there is no need for a "begin"
        method) """
        self.verbose = False
        self.begun = True

    def get_iteration_number(self):
        """ Called by AbstractOptimiser.optimise when initialising the iteration
        number """
        return 0

    def update(self, **kwargs):
        """ Called by AbstractOptimiser.optimise during and after the
        optimisation loop """
