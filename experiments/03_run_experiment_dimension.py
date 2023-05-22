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
expe_name = "dim_var_diff"
all_hyperparams[expe_name] = []
# all_dims = np.linspace(1, 125, 10).astype(int)
all_dims = np.linspace(1, 100, 10).astype(int)
all_path_names = ["geometric", "arithmetic", "arithmetic-adaptive", "arithmetic-adaptive-trig"]
all_losses = ["nce"]
all_vars = [1. / 4]

# outer loop
for dim, path_name, loss, var in product(
        all_dims, all_path_names, all_losses, all_vars):
    two_step = True if ("adaptive" in path_name) else False
    all_n_distributions = [2, 10] if path_name == "arithmetic-adaptive-trig" else [10]

    for n_distributions in all_n_distributions:
        target = MultivariateNormal(
            loc=torch.zeros(dim),
            covariance_matrix=var * torch.eye(dim)
        )
        target_logZ = ((dim / 2) * np.log(2 * np.pi) + (1. / 2) * torch.slogdet(target.covariance_matrix)[1]).item()
        # target_logZ = 0.
        hyperparams = {
            "dim": dim,
            "n_distributions": n_distributions,
            "path_name": path_name,
            "target": target,
            "target_logZ": target_logZ,
            "target_logZ_estim": target_logZ,
            "loss": loss,
            "two_step": two_step,
            "two_step_n_avg": 1,
            "two_step_path_name": "geometric",
            "two_step_n_distributions": 10,
            "two_step_loss": "nce"
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
        two_step = hyperparams["two_step"]
        two_step_loss = hyperparams["two_step_loss"]
        two_step_n_avg = hyperparams["two_step_n_avg"]
        two_step_path_name = hyperparams["two_step_path_name"]
        two_step_n_distributions = hyperparams["two_step_n_distributions"]

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
            loss=loss,
            two_step=two_step,
            two_step_loss=two_step_loss,
            two_step_n_avg=two_step_n_avg,
            two_step_path_name=two_step_path_name,
            two_step_n_distributions=two_step_n_distributions
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
