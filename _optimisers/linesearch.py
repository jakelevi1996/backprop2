import numpy as np

class LineSearch:
    def __init__(self, s0=1.0, alpha=0.5, beta=0.5, max_its=10):
        """ Initialise a LineSearch object. This object can be passed to a
        minimisation function (such as gradient_descent or generalised_newton),
        and to the initialiser for a Result object.

        Inputs:
        -   s0: initial step size, default is 1.0
        -   alpha: minimum ratio of reduction to expected reduction to accept,
            in (0, 1). A higher value of alpha means more steps will be taken
            per iteration when backtracking, to get a step size which gives a
            better reduction in error function. Default is 0.5. 
        -   beta: ratio to scale the step size by when backtracking (and inverse
            ratio to scale the step size by when forward tracking). Smaller beta
            means that the changes in step size will be bigger, so iterations
            will generally end faster, but the step size which is found might
            not be very optimal. Default is 0.5
        -   max_its: maximum number of iterations to perform every time
            get_step_size is called. Fewer iterations per function call might
            help to prevent the line-search from overfitting a certain batch,
            but will prevent finding the optimal step size every time
            get_step_size is called. Default is 10.
        """
        self.s = s0
        self.alpha = alpha
        self.beta = beta
        self.max_its = max_its
            
    def get_step_size(self, model, x, y, w, delta, dEdw):
        """
        Find the approximate locally best step size to use to optimise the model
        for the current batch of training data. This function is called by the
        minimise function (if a valid LineSearch object is passed to it).

        Inputs:
        -   model: instance of NeuralNetwork to be optimised
        -   x: inputs for the current batch of training data
        -   y: targets for the current batch of training data
        -   w: current parameters of the model, from which the step will be
            taken
        -   delta: direction in which to take a step
        -   dEdw: gradient of the error function at the current parameters
        """
        # Calculate initial parameters
        E_old = E_0 = model.mean_error(y)
        E_new = model(x, y, w + self.s * delta)
        delta_dot_dEdw = np.dot(delta, dEdw)

        # Check initial backtrack condition
        if self._must_backtrack(E_new, E_0, delta_dot_dEdw):
            # Reduce step size until reduction is good and stops decreasing
            for _ in range(self.max_its):
                self.s *= self.beta
                E_old = E_new
                E_new = model(x, y, w + self.s * delta)

                if (
                    (not self._must_backtrack(E_new, E_0, delta_dot_dEdw))
                    and E_new >= E_old
                ):
                    break

            if E_new > E_old:
                self.s /= self.beta
        else:
            # Track forwards until objective function stops decreasing
            for _ in range(self.max_its):
                self.s /= self.beta
                E_old = E_new
                E_new = model(x, y, w + self.s * delta)

                if E_new >= E_old:
                    break

            if E_new > E_old:
                self.s *= self.beta

        return self.s

    def _must_backtrack(self, E_new, E_0, delta_dot_dEdw):
        """ Compares the actual reduction in the objective function to that
        which is expected from a first-order Taylor expansion. Returns True if a
        reduction in the step size is needed according to this criteria,
        otherwise returns False. """
        reduction = E_0 - E_new
        expected_reduction = -self.s * delta_dot_dEdw
        return reduction < (self.alpha * expected_reduction)

# def check_bad_step_size(s):
#     if s == 0:
#         print("s has converged to 0; resetting t0 s_old * beta ...")
#         return True
#     elif not np.isfinite(s):
#         print("s has diverged; resetting t0 s_old/beta ...")
#         return True
#     else: return False
