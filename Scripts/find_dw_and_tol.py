"""
... 

TODO:
- Docstring
- Plotting
- Option for mean gradient
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
if __name__ == "__main__":
    import __init__
import data
from optimisers import results
from Tests.util import get_random_network_inputs_targets

def print_variable(label, value):
    print("{:30} = {:.3e}".format(label, value))

use_mean_gradient = False

plt.figure(figsize=[8, 6])

for dw_max_exp in range(-20, 0):
    # Set the maximum change in the weights for this iteration and print
    dw_max = 10 ** dw_max_exp
    print_variable("dw_max", dw_max)

    # Get network, inputs, targets, and gradient
    n, x, t, _ = get_random_network_inputs_targets(9011)
    grad_0 = n.get_gradient_vector(x, t).copy()

    # Change the weights and calculate the change in error function
    E_0 = n.mean_error(t, x)
    w = n.get_parameter_vector()
    dw = np.random.uniform(-dw_max, dw_max, grad_0.shape)
    n.set_parameter_vector(w + dw)
    E_1 = n.mean_error(t, x)
    dE = E_1 - E_0

    # Calculate the error in the approximation of the error-function and print
    error = abs(dE - np.dot(grad_0, dw))
    error_relative = abs(error / dE)
    print_variable("|E_0|", abs(E_0))
    print_variable("|dE|", abs(dE))
    print_variable("max(abs(gradient))", max(abs(grad_0)))
    print_variable("max(abs(gradient * dw))", max(abs(grad_0 * dw)))
    print_variable("Absolute gradient error", error)
    print_variable("***Relative gradient error***", error_relative)

    # Check the relative error after adding noise to the gradient
    mag = np.max(np.abs(grad_0))
    gradient_noisy = grad_0 + np.random.uniform(-mag, mag, grad_0.shape)
    error_noisy = abs((dE - np.dot(gradient_noisy, dw)) / dE)
    print_variable("Relative error with noise", error_noisy)

    # Get indices for Hessian blocks
    max_block_size = np.random.randint(3, 6)
    weight_inds_list = [
        np.array_split(
            np.random.permutation(layer.num_weights),
            np.ceil(layer.num_weights / max_block_size)
        ) for layer in n.layers
    ]
    bias_inds_list = [
        np.array_split(
            np.random.permutation(layer.num_bias),
            np.ceil(layer.num_bias / max_block_size)
        ) for layer in n.layers
    ]
    
    # Get initial parameters, gradient, and Hessian blocks
    w_0 = n.get_parameter_vector().copy()
    grad_0 = n.get_gradient_vector(x, t).copy()
    hess_block_list, hess_inds_list = n.get_hessian_blocks(
        x,
        t,
        weight_inds_list,
        bias_inds_list
    )
    error_list = []
    # Iterate through each block in the Hessian
    for hess_block, hess_inds in zip(hess_block_list, hess_inds_list):
        n.set_parameter_vector(w_0)
        dw = np.random.uniform(-dw_max, dw_max, len(hess_inds))
        w = n.get_parameter_vector()
        w[hess_inds] += dw
        n.set_parameter_vector(w)
        grad_1 = n.get_gradient_vector(x, t)
        d_grad = (grad_1 - grad_0)[hess_inds]
        relative_error = (d_grad - np.matmul(hess_block, dw)) / d_grad
        error_list.append(np.max(np.abs(relative_error)))
    error_relative_hessian = np.max(np.abs(error_list))
    print_variable("***Relative Hessian error***", error_relative_hessian)

    print("\n")

    # Add relative errors to plot
    plt.loglog(dw_max, error_relative, "bo", alpha=0.5)
    plt.loglog(dw_max, error_relative_hessian, "ro", alpha=0.5)

# Format, save and close the plot
name = "Relative errors as a function of maximum change in parameters"
plt.title(name)
plt.xlabel("$dw_{max}$")
plt.ylabel("Relative error in approximating the error function")
plt.legend(handles=[
    Line2D([], [], c="b", marker="o", ls="", label="Relative gradient error"),
    Line2D([], [], c="r", marker="o", ls="", label="Relative Hessian error")
])
plt.grid(True)
plt.tight_layout()
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Outputs")
plt.savefig(os.path.join(output_dir, name))
plt.close()
