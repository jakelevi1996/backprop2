class AbstractResult:
    """ The AbstractResult is used as an abstract parent class for
    _optimiser.results.Result, and is also used as a blank result by
    _optimisers.columns.OptimalBatchSize. This is to avoid the circular
    dependency of _optimiser.results needing to import columns so that it can
    have default columns, and _optimisers.columns needing to import results so
    that _optimisers.columns.OptimalBatchSize can use an empty result, which is
    more efficient than initiating a new result (and new column objects etc)
    every time the optimisation function is called """
    def __init__(self):
        """ Initialise an AbstractResult object, which by default is not
        verbose, and has already "begun" (so that there is no need for a "begin"
        method) """
        self.verbose = False
        self.begun = True

    def has_column_type(self, col_type):
        """ Called by _optimisers.minimise.minimise when determining the initial
        iteration number """
        return False

    def update(self, **kwargs):
        """ Called by _optimisers.minimise.minimise during and after the
        optimisation loop """
