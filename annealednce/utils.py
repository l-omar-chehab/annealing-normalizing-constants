"""Useful functions."""

import numpy as np
from scipy.linalg import circulant
import time


def parameter_distance(mean1, prec1, mean2, prec2):
    """Euclidean distance between the natural parameters of two gaussians,
     written as an exponential family."""
    distance_squared = np.linalg.norm(prec2 @ mean2 - prec1 @ mean1)**2 + \
        np.linalg.norm(0.5 * prec2 - 0.5 * prec1)**2
    return np.sqrt(distance_squared)


def gen_matrix_exp_decay(dim, decay):
    """Generate an identity matrix, where the off-diagonal entries are non-zero.
    Large decay: off-diagonal entries are near zero.
    Small decay: off-diagonal entries are near 1/2."""
    mat_expdecay = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(i):
            mat_expdecay[i, j] = i - j
    mat_expdecay = mat_expdecay + mat_expdecay.T
    mat_expdecay = np.exp(-decay * mat_expdecay)
    assert np.alltrue(np.linalg.eigvals(mat_expdecay) > 0)
    return mat_expdecay


def gen_frozen_random_invertible_matrix(dim, seed=0):
    """Generate a random matrix that is invertible."""
    rng = np.random.RandomState(seed)
    # Initialize matrix
    mat = np.zeros(shape=(dim, dim))
    # Repeat until the matrix is invertible
    while np.linalg.det(mat) == 0:   # this should be rare
        # Core matrix, 1x1
        mat = rng.randn(1).reshape(1, 1)
        # Add rows and columns until the dimension is reached
        for (len_col, len_row) in zip(range(1, dim), range(2, dim + 1)):
            mat = np.c_[mat, rng.randn(len_col)]  # add column
            mat = np.r_[mat, [rng.randn(len_row)]]  # add row
    return mat

# import numpy as np
# import matplotlib.pyplot as plt
# condnbs = []
# logdets = []
# for dim in range(1, 101):
#     mat = gen_frozen_random_invertible_matrix(dim, seed=0)
#     det = np.linalg.det(mat)
#     logdet = np.log(np.abs(det))
#     condnb = np.linalg.cond(mat)
#     print(f"dim: {dim} | conditioning: {condnb:.2f} | det: {det:.2f} | logdet: {logdet:.2f}")
#     condnbs.append(condnb)
#     logdets.append(logdet)

# fig, ax = plt.subplots()
# ax.scatter(condnbs, logdets)
# ax.set(
#     xlabel="Conditioning",
#     ylabel="Log-determinant",
#     xlim=(0, 100),
# )


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
            f"Iteration: {iteration} | Iterate: {iterate:.2f} | " \
            f"Loss: {loss:.2f} ({duration_loss:.2f} seconds)| Gradient: {grad:.2f} ({duration_grad:.2f} seconds)"
        )
    iteration += 1

    return False
