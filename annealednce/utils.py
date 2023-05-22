"""Useful functions."""

import numpy as np
import time


def parameter_distance(mean1, prec1, mean2, prec2):
    """Euclidean distance between the natural parameters of two gaussians,
     written as an exponential family."""
    distance_squared = np.linalg.norm(prec2 @ mean2 - prec1 @ mean1)**2 + \
        np.linalg.norm(0.5 * prec2 - 0.5 * prec1)**2
    return np.sqrt(distance_squared)


iteration = 0


def callback_scipy_minimize(
        xk,                                                      # scipy internal variable
        loss_func, grad_func,                                    # arguments added by me
        x_target, x_proposal, logf_target, logf_proposal,
        verbose=True
):
    """Callback function for scipy.optimize.minimize."""
    global iteration  # looks for a global variable (outside the function)

    # current iterate
    iterate = xk.item()

    # evaluate loss
    start_loss = time.time()
    loss = loss_func(xk, x_target, x_proposal, logf_target, logf_proposal)
    end_loss = time.time()
    duration_loss = end_loss - start_loss

    # evaluate gradient
    start_grad = time.time()
    grad = grad_func(xk, x_target, x_proposal, logf_target, logf_proposal)
    end_grad = time.time()
    duration_grad = end_grad - start_grad

    # print diagnostics
    if verbose:
        print(
            f"Iteration: {iteration} | Iterate: {iterate:.2f} | "
            f"Loss: {loss:.2f} ({duration_loss:.2f} seconds)| Gradient: {grad:.2f} ({duration_grad:.2f} seconds)"
        )
    iteration += 1

    return False
