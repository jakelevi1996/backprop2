from _optimisers.minimise import minimise, Result

def get_gradient_descent_step(model, dataset, learning_rate):
    """
    Method to get the descent step during each iteration of gradient-descent
    minimisation
    """
    dEdw = model.get_gradient_vector(dataset.x_train, dataset.y_train)

    return -learning_rate * dEdw, dEdw

def gradient_descent(
    model,
    dataset,
    learning_rate=1e-1,
    result=None,
    **kwargs
):
    """ TODO: why is this ~10% slower than the old SGD function? """
    get_step = lambda model, dataset: get_gradient_descent_step(
        model,
        dataset,
        learning_rate
    )
    
    if result is None:
        result = Result("Gradient descent")

    result = minimise(model, dataset, get_step, result=result, **kwargs)

    return result