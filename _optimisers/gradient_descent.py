from _optimisers.minimise import _minimise, Result

def get_gradient_descent_step(model, x_batch, y_batch, learning_rate):
    """ Method to get the descent step during each iteration of gradient-descent
    minimisation, using a learning rate """
    dEdw = model.get_gradient_vector(x_batch, y_batch)

    return -learning_rate * dEdw, dEdw

def get_gradient_descent_step_no_lr(model, x_batch, y_batch):
    """ Method to get the descent step during each iteration of gradient-descent
    minimisation, without a learning rate (this is intended to be used with a
    line-search) """
    dEdw = model.get_gradient_vector(x_batch, y_batch)

    return -dEdw, dEdw

def gradient_descent(
    model,
    dataset,
    learning_rate=1e-1,
    result=None,
    line_search=None,
    **kwargs
):
    """ TODO: why is this ~10% slower than the old SGD function? """
    if line_search is None:
        get_step = lambda model, x_batch, y_batch: get_gradient_descent_step(
            model,
            x_batch,
            y_batch,
            learning_rate
        )
    else:
        get_step = get_gradient_descent_step_no_lr
    
    if result is None:
        result = Result("Gradient descent")

    result = _minimise(
        model,
        dataset,
        get_step,
        result=result,
        line_search=line_search,
        **kwargs
    )

    return result
