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

DIM = 50

# Experiment: gaussian target
expe_name = "param_distance"
all_hyperparams[expe_name] = []
all_dims = [50]
all_path_names = [
    "geometric", "arithmetic", "arithmetic-adaptive", "arithmetic-adaptive-trig"]
all_variances = 1. / np.linspace(1.05, 3, 10)**2

# outer loop
for (path_name, variance) in product(all_path_names, all_variances):

    # Compute variables which depend on the choice of path
    two_step = True if ("adaptive" in path_name) else False
    all_n_distributions = [2, 10] if path_name == "arithmetic-adaptive-trig" else [10]

    for n_distributions in all_n_distributions:
        # Compute parameters of the target
        mean = np.zeros(DIM)
        cov = variance * np.eye(DIM)
        prec = np.linalg.inv(cov)
        # Compute parameter distance
        param_distance = parameter_distance(mean1=np.zeros(DIM), prec1=np.eye(DIM), mean2=mean, prec2=prec)
        # Instantiate target (torch)
        target = MultivariateNormal(
            loc=torch.from_numpy(mean),
            covariance_matrix=torch.from_numpy(cov))
        # Compute target logZ
        target_logZ_canonic = ((DIM / 2) * np.log(2 * np.pi) + (1. / 2) * torch.slogdet(target.covariance_matrix)[1]).item()
        all_target_logZs = [target_logZ_canonic]
        for target_logZ in all_target_logZs:
            hyperparams = {
                "dim": DIM,
                "n_distributions": n_distributions,
                "path_name": path_name,
                "target": target,
                "param_distance": param_distance,
                "target_logZ": target_logZ,
                "target_logZ_estim": target_logZ,
                "loss": "nce",
                "two_step": two_step,
                "two_step_n_avg": 1,
                "two_step_path_name": "geometric",
                "two_step_n_distributions": 10,
                "two_step_loss": "nce"
            }
            all_hyperparams[expe_name].append(hyperparams)


# %%
# # Create a dictionary with all hyperparameter combinations
# all_hyperparams = {}


# # Experiment: gaussian target
# expe_name = "param_distance"
# all_hyperparams[expe_name] = []
# all_dims = [50]
# all_path_names = [
#     "geometric", "arithmetic", "arithmetic-adaptive", "arithmetic-adaptive-trig"]
# all_losses = ["nce"]
# N_PARAMS = 5

# # outer loop
# for dim, path_name, loss in product(
#         all_dims, all_path_names, all_losses):
#     two_step = True if ("adaptive" in path_name) else False
#     # inner loop: parameter range and target logZ depend on dim
#     # if dim == 10:
#     #     # max_scale = 14
#     #     max_scale = 25
#     # elif dim == 2:
#     #     max_scale = 25  # 10

#     # all_mean_norms = 0. * np.linspace(0.1, max_scale, N_PARAMS)
#     # all_cov_decays = 1. / np.linspace(0.1, max_scale, N_PARAMS)
#     indices = np.linspace(1.05, 3, N_PARAMS)

#     # for (mean_norm, cov_decay) in zip(all_mean_norms, all_cov_decays):
#     for idx in indices:
#         # Define target parameters (numpy)
#         # mean = mean_norm * np.ones(dim) / np.linalg.norm(np.ones(dim))
#         mean = np.zeros(dim)
#         # cov = gen_matrix_exp_decay(dim=dim, decay=cov_decay)
#         # mean = idx**2 * np.ones(dim) / np.linalg.norm(np.ones(dim))
#         cov = np.eye(dim) / (idx**2)
#         prec = np.linalg.inv(cov)
#         # Compute parameter distance
#         param_distance = parameter_distance(mean1=np.zeros(dim), prec1=np.eye(dim), mean2=mean, prec2=prec)
#         # Instantiate target (torch)
#         target = MultivariateNormal(
#             loc=torch.from_numpy(mean),
#             covariance_matrix=torch.from_numpy(cov))
#         # Compute target logZ
#         target_logZ_canonic = ((dim / 2) * np.log(2 * np.pi) + (1. / 2) * torch.slogdet(target.covariance_matrix)[1]).item()
#         # all_target_logZs = [0.] if path_name == "geometric" else [0., target_logZ_canonic]
#         # all_target_logZs = [4.]
#         all_target_logZs = [target_logZ_canonic]
#         for target_logZ in all_target_logZs:
#             hyperparams = {
#                 "dim": dim,
#                 "n_distributions": n_distributions,
#                 "path_name": path_name,
#                 "target": target,
#                 "param_distance": param_distance,
#                 "target_logZ": target_logZ,
#                 "target_logZ_estim": target_logZ,
#                 "loss": loss,
#                 "two_step": two_step
#             }
#             all_hyperparams[expe_name].append(hyperparams)

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
            verbose=1,
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
