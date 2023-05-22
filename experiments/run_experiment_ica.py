"""Run experiments on Annealed NCE."""

# %%
import torch
from torch.distributions import MultivariateNormal
import numpy as np
from itertools import product
import time

from annealednce.nce import annealed_nce_estim
from annealednce.distributions import GeneralizedMultivariateNormal, DistributionICA
from annealednce.glow_sandbox import GlowCeleba
from annealednce.glow_model import Glow
from annealednce.defaults import RESULTS_FOLDER
from annealednce.utils import parameter_distance, gen_matrix_exp_decay


# %%
# Define global variables
N_JOBS = 100
SAMPLE_SIZE = 50000
N_REPEATS = 100

# %%
# Create a dictionary with all hyperparameter combinations
all_hyperparams = {}


# Experiment: ica target
expe_name = "ica_target"
all_hyperparams[expe_name] = []
all_dims = [50]
# all_bias_norms = np.linspace(0, 20, 21).astype(int)
all_n_distributions = [2, 3]   # np.arange(2, 11).astype(int)
all_path_names = ["arithmetic-schedule"]
all_losses = ["nce"]

# outer loop
for dim, n_distributions, path_name, loss in product(
        all_dims, all_n_distributions, all_path_names, all_losses):
    target = DistributionICA(dim=dim, bias_norm=0.)
    # Compute target logZ
    target_logZ_canonic = target.logZ_canonical
    all_target_logZs = [target_logZ_canonic]
    for target_logZ in all_target_logZs:
        hyperparams = {
            "dim": dim,
            "n_distributions": n_distributions,
            "path_name": path_name,
            "target": target,
            "target_logZ": target_logZ,
            "target_logZ_estim": target_logZ,
            "loss": loss,
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
            verbose=True,
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
