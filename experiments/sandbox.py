# %%
import numpy as np
from scipy.linalg import circulant
import matplotlib.pyplot as plt


DIM = 50


# %%
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
    mat_expdecay = np.zeros((DIM, DIM))
    for i in range(dim):
        for j in range(i):
            mat_expdecay[i, j] = i - j
    mat_expdecay = mat_expdecay + mat_expdecay.T
    mat_expdecay = np.exp(-decay * mat_expdecay)
    assert np.alltrue(np.linalg.eigvals(mat_expdecay) > 0)
    return mat_expdecay


# %%
mean1, prec1 = np.zeros(DIM), np.eye(DIM)

indices = np.linspace(0.1, 14, 10)
means = []
covs = []
lognormalizations = []

distances = []
for idx in indices:
    # Compute the mean of the second gaussian
    ones = np.ones(DIM)
    mean2 = idx * ones / np.linalg.norm(ones)
    means.append(mean2)
    # Compute the precision of the second gaussian
    cov2 = gen_matrix_exp_decay(dim=DIM, decay=1. / idx)
    covs.append(cov2)
    prec2 = np.linalg.inv(cov2)
    # Compute the distance between the two gaussian
    distance = parameter_distance(mean1, prec1, mean2, prec2)
    distances.append(distance)
    # Compute the lognormalization of the target
    lognormalization = (DIM / 2) * np.log(2 * np.pi) + (1. / 2) * np.log(np.abs(np.linalg.det(cov2)))
    lognormalizations.append(lognormalization)
plt.plot(indices, distances)
# plt.plot(indices, covs)

# %%
# %%
mean1, prec1 = np.zeros(DIM), np.eye(DIM)

# indices = np.linspace(0.1, 20, 10)
indices = np.linspace(1.05, 2.5, 4)
distances = []
lognormalizations = []

for idx in indices:
    # Compute the mean of the second gaussian
    # mean2 = idx**2 * np.ones(DIM) / np.linalg.norm(ones)
    mean2 = np.zeros(DIM)
    # Compute the precision of the second gaussian
    cov2 = np.eye(DIM) / idx**2
    prec2 = np.linalg.inv(cov2)
    # Compute the distance between the two gaussian
    distance = parameter_distance(mean1, prec1, mean2, prec2)
    distances.append(distance)
    # Compute the log-normalization of the target
    lognormalization = (DIM / 2) * np.log(2 * np.pi) + (1. / 2) * np.log(np.abs(np.linalg.det(cov2)))
    lognormalizations.append(lognormalization)

plt.plot(indices, distances, label="distance")
plt.plot(indices, lognormalizations, label="lognormalization")
plt.legend()

# %%
# %%

all_dims = np.linspace(1, 50, 5).astype(int)
distances = []
lognormalizations = []

for dim in all_dims:
    # Compute the mean and precision of the first gaussian
    mean1 = np.zeros(dim)
    prec1 = np.eye(dim)
    # Compute the mean of the second gaussian
    mean2 = 5 * np.ones(dim)
    # Compute the precision of the second gaussian
    cov2 = 1 * np.eye(dim)
    prec2 = np.linalg.inv(cov2)
    # Compute the distance between the two gaussian
    distance = parameter_distance(mean1, prec1, mean2, prec2)
    distances.append(distance)
    # Compute the log-normalization of the target
    lognormalization = (dim / 2) * np.log(2 * np.pi) + (1. / 2) * np.log(np.abs(np.linalg.det(cov2)))
    lognormalizations.append(lognormalization)

plt.plot(all_dims, distances, label="distance")
plt.plot(all_dims, lognormalizations, label="lognormalization")
plt.legend()

# %%
