import numpy as np

def backtrack_condition(s, E_new, E_0, delta_dot_dEdw, alpha):
    """
    Compares the actual reduction in the objective function to that which is
    expected from a first-order Taylor expansion. Returns True if a reduction in
    the step size is needed according to this criteria, otherwise returns False.

    TODO: try f_new > f_0 - (alpha * expected_reduction)
    """
    reduction = E_0 - E_new
    # if reduction == 0: return False
    expected_reduction = -s * delta_dot_dEdw
    return reduction < (alpha * expected_reduction)

def check_bad_step_size(s):
    if s == 0:
        print("s has converged to 0; resetting t0 s_old * beta ...")
        return True
    elif not np.isfinite(s):
        print("s has diverged; resetting t0 s_old/beta ...")
        return True
    else: return False

def line_search(model, x, y, w, s, delta, dEdw, alpha, beta):
    """
    ... TODO: use batches, and implement maximum number of steps of line search,
    using a for-loop with an `if condition: break` statement
    """
    # Calculate initial parameters
    E_0 = model.mean_error(y, x)
    E_old = E_0
    model.set_parameter_vector(w + s * delta)
    E_new = model.mean_error(y, x)
    
    delta_dot_dEdw = np.dot(delta, dEdw)
    bt_params = (E_0, delta_dot_dEdw, alpha)

    # Check initial backtrack condition
    if backtrack_condition(s, E_new, *bt_params):
        # Reduce step size until reduction is good enough and stops decreasing
        s *= beta
        E_old = E_new
        model.set_parameter_vector(w + s * delta)
        E_new = model.mean_error(y, x)

        while backtrack_condition(s, E_new, *bt_params) or E_new < E_old:
            s *= beta
            E_old = E_new
            model.set_parameter_vector(w + s * delta)
            E_new = model.mean_error(y, x)
        if E_new > E_old:
            s /= beta
    else:
        # Track forwards until objective function stops decreasing
        s /= beta
        E_old = E_new
        model.set_parameter_vector(w + s * delta)
        E_new = model.mean_error(y, x)
        while E_new < E_old:
            s /= beta
            E_old = E_new
            model.set_parameter_vector(w + s * delta)
            E_new = model.mean_error(y, x)
        if E_new > E_old:
            s *= beta

    return s
