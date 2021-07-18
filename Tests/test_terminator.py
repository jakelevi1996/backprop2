""" Unit tests for the classes in the optimisers.terminator module """

import os
import numpy as np
import pytest
import optimisers
import models
import data
from .util import (
    get_random_network,
    get_random_dataset,
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

@pytest.mark.parametrize("repeat", [1, 2, 3])
def test_dynamic_terminator(repeat):
    set_random_seed_from_args("test_dynamic_terminator", repeat)
    num_iters = np.random.randint(20, 50)
    dataset = get_random_dataset()
    model = get_random_network(
        input_dim=dataset.input_dim,
        output_dim=dataset.output_dim,
    )
    batch_size = np.random.randint(3, dataset.n_train)
    dynamic_terminator = optimisers.DynamicTerminator(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        i_lim=num_iters,
    )
    dynamic_terminator.set_initial_iteration(0)
    for i in range(num_iters):
        x_batch, y_batch = dynamic_terminator.get_batch(dataset)
        assert x_batch.shape == (dataset.input_dim, batch_size)
        assert y_batch.shape == (dataset.output_dim, batch_size)
        assert len(dynamic_terminator._p_improve_list) == (i + 1)
    
    assert dynamic_terminator.ready_to_terminate(num_iters)
