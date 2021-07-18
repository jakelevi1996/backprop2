""" Unit tests for the classes in the optimisers.terminator module """

import os
import numpy as np
import pytest
import optimisers
import models
import data
from .util import (
    get_random_network_inputs_targets,
    get_output_dir,
    set_random_seed_from_args,
)

# Get name of output directory, and create it if it doesn't exist
output_dir = get_output_dir("Terminator")

@pytest.mark.parametrize("repeat", [1, 2, 3])
def test_terminator(repeat):
    set_random_seed_from_args("test_terminator", repeat)
    num_iters = np.random.randint(20, 50)
    terminator = optimisers.Terminator(i_lim=num_iters)
    terminator.set_initial_iteration(0)
    for i in range(num_iters):
        assert not terminator.ready_to_terminate(i)
    
    assert terminator.ready_to_terminate(num_iters)
