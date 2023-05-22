"""Run experiments on Annealed NCE."""

# %%
import torch
from torch.distributions import MultivariateNormal
import numpy as np
from itertools import product
import time

from annealednce.nce import annealed_nce_estim
from annealednce.defaults import RESULTS_FOLDER


# %%
# Define global variables
N_JOBS = 100
SAMPLE_SIZE = 50000
N_REPEATS = 100
VERBOSE = False

# %%
# Create a dictionary with all hyperparameter combinations
all_hyperparams = {}


# Experiment: gaussian target
expe_name = "loss"
all_hyperparams[expe_name] = []
all_dims = [50]
all_n_distributions = [2, 3, 5]
all_path_names = ["geometric"]
all_losses = ["exp", "rev-is", "is", "nce"]
all_variances = [2.]

# outer loop
for dim, n_distributions, path_name, loss, variance in product(
        all_dims, all_n_distributions, all_path_names, all_losses, all_variances):
    # target = GeneralizedMultivariateNormal(
    #     means=[0] * dim, variances=[1.] * dim, powers=[power] * dim)
    target = MultivariateNormal(
        loc=torch.zeros(dim), covariance_matrix=variance * torch.eye(dim))
    # target_logZ = ((dim / 2) * np.log(2 * np.pi) + (1. / 2) * torch.slogdet(target.covariance_matrix)[1]).item()
    target_logZ = 0.
    hyperparams = {
        "dim": dim,
        "n_distributions": n_distributions,
        "path_name": path_name,
        # "power": power,
        # "mean": mean,
        "target": target,
        "target_logZ": target_logZ,
        "target_logZ_estim": target_logZ,
        "loss": loss,
        "variance": variance
    }
    all_hyperparams[expe_name].append(hyperparams)


# %%
# Run experiments
expe_names = list(all_hyperparams.keys())

for expe_name in expe_names:

    print("Starting experiment: ", expe_name)
    start = time.time()

    results = []
    for hyperparams in all_hyperparams[expe_name]:

        # Unpack hyperparameters
        dim = hyperparams["dim"]
        n_distributions = hyperparams["n_distributions"]
        path_name = hyperparams["path_name"]
        target = hyperparams["target"]
        target_logZ = hyperparams["target_logZ"]
        target_logZ_estim = hyperparams["target_logZ_estim"]
        loss = hyperparams["loss"]

        # Estimation
        beta_hats, error, error_best = annealed_nce_estim(
            proposal=MultivariateNormal(loc=torch.zeros(dim), covariance_matrix=torch.eye(dim)),
            target=target,
            target_logZ=target_logZ,
            target_logZ_estim=target_logZ_estim,
            path_name=path_name,
            n_distributions=n_distributions,
            n_repeats=N_REPEATS,
            n_jobs=N_JOBS,
            sample_size=SAMPLE_SIZE,
            verbose=VERBOSE,
            loss=loss
        )

        # Record result
        result = {**hyperparams, "error": error,
                  "error_best": error_best, "logZs": beta_hats}
        results.append(result)

    end = time.time()
    duration_in_minutes = np.round((end - start) / 60, 1)

    filename = RESULTS_FOLDER / f"annealed_nce_expe_{expe_name}.th"
    torch.save(results, filename)

    print(f"Finished experiment {expe_name} in {duration_in_minutes} minutes.")

# %%
