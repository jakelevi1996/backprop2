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

