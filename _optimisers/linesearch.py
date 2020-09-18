import numpy as np

class LineSearch:
    def __init__(self, s0=1.0, alpha=0.5, beta=0.5):
        self.s = s0
        self.alpha = alpha
        self.beta = beta
            
    def get_step_size(self, model, x, y, w, delta, dEdw):
        """
        ... TODO: use batches, and implement maximum number of steps of line
        search, using a for-loop with an `if condition: break` statement
        """
        # Calculate initial parameters
        E_0 = model.mean_error(y, x)
        E_old = E_0
        model.set_parameter_vector(w + self.s * delta)
        E_new = model.mean_error(y, x)
        
        delta_dot_dEdw = np.dot(delta, dEdw)
        bt_params = (E_0, delta_dot_dEdw)

        # Check initial backtrack condition
        if self._backtrack_condition(E_new, *bt_params):
            # Reduce step size until reduction is good and stops decreasing
            self.s *= self.beta
            E_old = E_new
            model.set_parameter_vector(w + self.s * delta)
            E_new = model.mean_error(y, x)

            while self._backtrack_condition(E_new, *bt_params) or E_new < E_old:
                self.s *= self.beta
                E_old = E_new
                model.set_parameter_vector(w + self.s * delta)
                E_new = model.mean_error(y, x)
            if E_new > E_old:
                self.s /= self.beta
        else:
            # Track forwards until objective function stops decreasing
            self.s /= self.beta
            E_old = E_new
            model.set_parameter_vector(w + self.s * delta)
            E_new = model.mean_error(y, x)
            while E_new < E_old:
                self.s /= self.beta
                E_old = E_new
                model.set_parameter_vector(w + self.s * delta)
                E_new = model.mean_error(y, x)
            if E_new > E_old:
                self.s *= self.beta

        return self.s

    def _backtrack_condition(self, E_new, E_0, delta_dot_dEdw):
        """
        Compares the actual reduction in the objective function to that which is
        expected from a first-order Taylor expansion. Returns True if a
        reduction in the step size is needed according to this criteria,
        otherwise returns False.
        """
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
