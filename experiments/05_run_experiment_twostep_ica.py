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
SAMPLE_SIZE = 50000
N_REPEATS = 100
VERBOSE = False

# %%
# Create a dictionary with all hyperparameter combinations
all_hyperparams = {}


# Experiment: gaussian target
expe_name = "twostep_ica"
all_hyperparams[expe_name] = []
all_path_names = ["arithmetic-adaptive", "arithmetic-adaptive-trig"]
all_n_distributions = [2, 20]

# outer loop
for path_name, n_distributions in product(all_path_names, all_n_distributions):
    two_step = True if ("adaptive" in path_name) else False
    # target = GeneralizedMultivariateNormal(
    #     means=[0] * dim, variances=[1.] * dim, powers=[power] * dim)
    # target = MultivariateNormal(
    #     loc=5 * torch.ones(2), covariance_matrix=torch.eye(2))
    # target_logZ = 10.
    target = DistributionICA(dim=50, bias_norm=0.)
    target_logZ = target.logZ_canonical
    hyperparams = {
        "dim": 50,
        "n_distributions": n_distributions,
        "path_name": path_name,
        # "power": power,
        # "mean": mean,
        "two_step": two_step,
        "two_step_n_avg": 10,
        "two_step_path_name": "arithmetic",
        "two_step_n_distributions": 2,
        "two_step_loss": "nce",
        "target": target,
        "target_logZ": target_logZ,
        "target_logZ_estim": target_logZ,
        "loss": "nce",
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

# %%
