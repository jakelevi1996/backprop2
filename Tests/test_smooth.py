""" Module containing tests for classes in the optimisers.smooth module """

import numpy as np
import pytest
from optimisers import smooth
import plotting
from .util import set_random_seed_from_args, get_output_dir

# Get name of output directory, and create it if it doesn't exist
output_dir = get_output_dir("Smooth")

@pytest.mark.parametrize("smoother_type", smooth.smoother_dict.values())
def test_plot_smoother(smoother_type):
    """ Given a type of smoother (which should be a subclass of the _Smoother
    abstract parent class), test initialising that smoother using default
    parameters, and using the smoother to smooth some simple noisy sinusoidal
    data, and plot the results """
    set_random_seed_from_args("test_plot_smoother")
    t = np.linspace(0, 3, 100)
    x = np.sin(2 * np.pi * t) + 0.1*np.random.normal(size=t.shape)
    smoother = smoother_type(x[0])
    y = [smoother.smooth(xi) for xi in x]
    plotting.simple_plot(
        [t, x, "bo-", t, y, "r--"],
        "test_plot_smoother(%s)" % smoother_type.__name__,
        output_dir,
        legend_kwarg_list=[
            {"label": "Input signal", "c": "b", "ls": "-", "marker": "o"},
            {"label": "Smoothed signal", "c": "r", "ls": "--"},
        ],
    )

@pytest.mark.parametrize("smoother_type", [
    smooth.Identity,
    smooth.Exponential,
    smooth.MovingAverage,
    smooth.MovingMaximum,
])
def test_smoother_constant_input_output(smoother_type):
    """ Test that, for appropriate types of smoothers, a constant input signal
    gives a constant output signal over the course of multiple data points, and
    that after the constant input signal, a change in the input signal causes a
    change in the output signal """
    set_random_seed_from_args("test_plot_smoother", smoother_type)
    x0 = np.random.normal()
    n = np.random.randint(50, 100)
    smoother = smoother_type(x0)
    for _ in range(n):
        y = smoother.smooth(x0)
        assert y == x0
    
    x1 = x0 + 1
    y = smoother.smooth(x1)
    assert y != x0

def test_moving_average_parameters():
    """ Test moving average filters with a range of buffer lengths """
    set_random_seed_from_args("test_smoother_parameters")
    t = np.linspace(0, 3, 100)
    x = np.sin(2 * np.pi * t) + 0.5*np.random.normal(size=t.shape)
    y_list = [
        [
            smoother.smooth(xi)
            for xi in x
        ]
        for n in range(5, 25, 2)
        for smoother in [smooth.MovingAverage(x[0], n=n)]
    ]
    plotting.simple_plot(
        [t, x, "bo-"] + [i for y in y_list for i in [t, y, "r-"]],
        "test_moving_average_parameters",
        output_dir,
        legend_kwarg_list=[
            {"label": "Input signal", "c": "b", "ls": "-", "marker": "o"},
            {"label": "Smoothed signal, $5\\leq n<25$", "c": "r", "ls": "-"},
        ],
    )

def test_exponential_parameters():
    """ Test exponential smoothers with a range of values for alpha (smoothing
    constant) """
    set_random_seed_from_args("test_smoother_parameters")
    t = np.linspace(0, 3, 100)
    x = np.sin(2 * np.pi * t) + 0.5*np.random.normal(size=t.shape)
    y_list = [
        [
            smoother.smooth(xi)
            for xi in x
        ]
        for alpha in np.linspace(0.1, 0.5, 10)
        for smoother in [smooth.Exponential(x[0], alpha=alpha)]
    ]
    plotting.simple_plot(
        [t, x, "bo-"] + [i for y in y_list for i in [t, y, "r-"]],
        "test_exponential_parameters",
        output_dir,
        legend_kwarg_list=[
            {"label": "Input signal", "c": "b", "ls": "-", "marker": "o"},
            {
                "label": "Smoothed signal, $0.1\\leq \\alpha<0.5$",
                "c": "r",
                "ls": "-",
            },
        ],
    )
