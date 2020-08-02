import numpy as np
from models import NeuralNetwork
import activations as a
import errors as e

def get_random_network(
    low=3,
    high=6,
    act_funcs=None,
    error_func=None
):
    """
    Generate a neural network with a random number of inputs, outputs, hidden
    layers, and number of units in each hidden layer
    """
    input_dim = np.random.randint(low, high)
    output_dim = np.random.randint(low, high)
    num_hidden_layers = np.random.randint(low, high)
    num_hidden_units = np.random.randint(low, high, num_hidden_layers)
    n = NeuralNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        num_hidden_units=num_hidden_units,
        act_funcs=act_funcs,
        error_func=error_func
    )
    return n

def get_random_inputs(input_dim, N_D_low=5, N_D_high=15):
    """ Generate random input data, with a random number of data points """
    N_D = np.random.randint(N_D_low, N_D_high)
    x = np.random.normal(size=[input_dim, N_D])
    return x, N_D

def get_random_targets(output_dim, N_D):
    """ Generate random target data """
    t = np.random.normal(size=[output_dim, N_D])
    return t

def get_random_network_inputs_targets(
    low=3,
    high=6,
    N_D_low=5,
    N_D_high=15,
    act_funcs=None,
    error_func=None
):
    """
    Wrapper for the following functions: get_random_network, get_random_inputs,
    get_random_targets
    """
    n = get_random_network(low, high, act_funcs, error_func)
    x, N_D = get_random_inputs(n.input_dim, N_D_low, N_D_high)
    t = get_random_targets(n.output_dim, N_D)
    return n, x, t, N_D

def iterate_random_seeds(*seeds):
    """
    This function can be used to return a decorator, which will automatically
    repeat a test function multiple times with different random seeds (the seeds
    are provided as arguments to this function). It is assumed that the function
    being decorated accepts no arguments, and returns no values (minor
    modifications would be needed if these assumptions were untrue). The
    decorator can be used as follows:

    ```
    @iterate_random_seeds(5920, 2788, 235)
    def function_name():
        do_function_body()
    ```
    """
    # decorator_func is the decorator which is returned, given the seeds
    def decorator_func(func):
        # func_wrapper is called when the decorated function is called
        def func_wrapper():
            # Call decorated function once with each random seed
            for s in seeds:
                np.random.seed(s)
                func()

        # Calling the decorator returns the decorated function wrapper
        return func_wrapper

    # When this function is called, the decorator is returned
    return decorator_func

def generate_decorator_expression(num_expressions=10):
    """
    This function can be used to print multiple decorator expressions for the
    decorator above, with different input random seeds
    """
    for _ in range(num_expressions):
        print("@iterate_random_seeds({}, {}, {})".format(
            *np.random.randint(0, 10000, size=[3])
        ))
