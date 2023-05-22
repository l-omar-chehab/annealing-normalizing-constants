"""Run experiments on Annealed NCE."""

# %%
import torch
from torch.distributions import MultivariateNormal
import numpy as np
from itertools import product
import time

from annealednce.distributions import DistributionICA
from annealednce.nce import annealed_nce_estim
from annealednce.defaults import RESULTS_FOLDER


# %%
# Define global variables
N_JOBS = 100
SAMPLE_SIZE = 200000  # 50000
N_REPEATS = 100
VERBOSE = False

# %%
# Create a dictionary with all hyperparameter combinations
all_hyperparams = {}


# Experiment: gaussian target
expe_name = "twostep_gaussian_lowdim"
all_hyperparams[expe_name] = []
all_path_names = ["geometric", "arithmetic", "arithmetic-adaptive", "arithmetic-adaptive-trig"]
DIM = 10

# generate random covariance matrix for target distribution
is_pos_def = False
while ~is_pos_def:
    cov_sqrt = np.random.rand(DIM, DIM)
    cov = np.dot(cov_sqrt, cov_sqrt.transpose())
    cov = torch.from_numpy(cov).float()
    is_pos_def = torch.all(torch.linalg.eig(cov)[0].real > 0)
target = MultivariateNormal(
    loc=torch.zeros(DIM), covariance_matrix=cov)
target_logZ = ((DIM / 2) * np.log(2 * np.pi) + (1. / 2) * torch.slogdet(target.covariance_matrix)[1]).item()

# outer loop
for path_name in all_path_names:
    two_step = True if ("adaptive" in path_name) else False
    all_n_distributions = [2, 10] if path_name == "arithmetic-adaptive-trig" else [10]

    for n_distributions in all_n_distributions:
        hyperparams = {
            "dim": DIM,
            "n_distributions": n_distributions,
            "path_name": path_name,
            "two_step": two_step,
            "two_step_n_avg": 1,
            "two_step_path_name": "geometric",
            "two_step_n_distributions": 10,
            "two_step_loss": "is",
            "target": target,
            "target_logZ": target_logZ,
            "target_logZ_estim": target_logZ,
            "loss": "is"
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
            two_step=two_step
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
