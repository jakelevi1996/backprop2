from time import perf_counter

class Result():
    """
    Class to store the results of optimisation in a single object which can be
    passed directly to plotting and/or analysis functions. Also contains methods
    for updating and displaying results

    TODO:
    - Make this class configurable, so columns such as step-size and |x| are
    optional, and the column width and format spec for each column is
    configurable
    - Implement saving and loading of results
    """
    def __init__(self, name=None, verbose=True):
        """
        Store the name of the experiment (which is useful later when displaying
        results), display table headers, initialise lists for objective function
        evaluations and the time and iteration number for each evaluation, and
        record the start time for the results list
        """
        self.name = name if (name is not None) else "Unnamed experiment"
        if verbose: self.display_headers()
        self.verbose = verbose

        self.train_errors   = []
        self.test_errors    = []
        self.times          = []
        self.iters          = []
        self.step_size      = []
        self.start_time     = perf_counter()
        # TODO: DBS criterion
    
    def time_elapsed(self): return perf_counter() - self.start_time
    
    def update(self, model, dataset, i, s):
        t = self.time_elapsed()
        e_train = model.mean_error(dataset.y_train, dataset.x_train)
        e_test  = model.mean_error(dataset.y_test, dataset.x_test)
        self.train_errors.append(e_train)
        self.test_errors.append(e_test)
        self.times.append(t)
        self.iters.append(i)
        self.step_size.append(s)
        if self.verbose: self.display_last()
    
    def display_headers(self):
        # num_fields, field_width = 3, 10
        print("\nPerforming test \"{}\"...".format(self.name))
        print("{:9} | {:8} | {:11} | {:11} | {:10}".format(
            "Iteration", "Time (s)", "Train error", "Test error", "Step size"))
        print(" | ".join("-" * i for i in [9, 8, 11, 11, 10]))

    def display_last(self):
        print("{:9d} | {:8.3f} | {:11.5f} | {:11.5f} | {:10.4f}".format(
            self.iters[-1], self.times[-1], self.train_errors[-1],
            self.test_errors[-1], self.step_size[-1]))

    def display_summary(self, n_iters):
        t_total = self.time_elapsed()
        t_mean = t_total / n_iters
        print("-" * 50,
            "{:30} = {}".format("Test name", self.name),
            "{:30} = {:,.4f} s".format("Total time", t_total),
            "{:30} = {:,}".format("Total iterations", n_iters),
            "{:30} = {:.4f} ms".format("Average time per iteration",
                1e3 * t_mean),
            "{:30} = {:,.1f}".format("Average iterations per second",
                1 / t_mean),
            sep="\n", end="\n\n")
    
    def save(self, filename): raise NotImplementedError
    def load(self, filename): raise NotImplementedError
