"""NCE estimator"""

# %%
import numpy as np
from functools import partial
from scipy.optimize import minimize
import torch
from torch.distributions import MultivariateNormal
from itertools import product
from joblib.parallel import Parallel, delayed

from annealednce.distributions import UnnormalizedDistributionPath
from annealednce.utils import callback_scipy_minimize


# %%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logistic_loss(beta, x_target, x_proposal, logf_target, logf_proposal):
    """Logistic classification loss."""
    target_term = -np.log(sigmoid(logf_target(x_target) - logf_proposal(x_target) - beta)).mean()
    proposal_term = -np.log(1 - sigmoid(logf_target(x_proposal) - logf_proposal(x_proposal) - beta)).mean()
    return (target_term + proposal_term).item()


def grad_logistic_loss(beta, x_target, x_proposal, logf_target, logf_proposal):
    """Gradient of the logistic classification loss."""
    target_term = (1 - sigmoid(logf_target(x_target) - logf_proposal(x_target) - beta)).mean()
    proposal_term = -sigmoid(logf_target(x_proposal) - logf_proposal(x_proposal) - beta).mean()
    return (target_term + proposal_term).item()


def is_loss(beta, x_target, x_proposal, logf_target, logf_proposal):
    """Importance Sampling classification loss."""
    target_term = beta
    proposal_term = np.exp(-beta) * np.exp(logf_target(x_proposal) - logf_proposal(x_proposal)).mean()
    return (target_term + proposal_term).item()


def grad_is_loss(beta, x_target, x_proposal, logf_target, logf_proposal):
    """Gradient of the Importance Sampling classification loss."""
    target_term = 1
    proposal_term = -np.exp(-beta) * np.exp(logf_target(x_proposal) - logf_proposal(x_proposal)).mean()
    return (target_term + proposal_term).item()


def revis_loss(beta, x_target, x_proposal, logf_target, logf_proposal):
    """Reverse Importance Sampling classification loss."""
    target_term = np.exp(beta) * np.exp(logf_proposal(x_target) - logf_target(x_target)).mean()
    proposal_term = -beta
    return (target_term + proposal_term).item()


def grad_revis_loss(beta, x_target, x_proposal, logf_target, logf_proposal):
    """Gradient of the Reverse Importance Sampling classification loss."""
    target_term = np.exp(beta) * np.exp(logf_proposal(x_target) - logf_target(x_target)).mean()
    proposal_term = -1
    return (target_term + proposal_term).item()


def exp_loss(beta, x_target, x_proposal, logf_target, logf_proposal):
    """Exponential classification loss."""
    target_term = np.exp(beta / 2) * np.exp(0.5 * logf_proposal(x_target) - 0.5 * logf_target(x_target)).mean()
    proposal_term = np.exp(-beta / 2) * np.exp(0.5 * logf_target(x_proposal) - 0.5 * logf_proposal(x_proposal)).mean()
    return (target_term + proposal_term).item()


def grad_exp_loss(beta, x_target, x_proposal, logf_target, logf_proposal):
    """Gradient of the exponential classification loss."""
    target_term = 0.5 * np.exp(beta / 2) * np.exp(0.5 * logf_proposal(x_target) - 0.5 * logf_target(x_target)).mean()
    proposal_term = -0.5 * np.exp(-beta / 2) * np.exp(0.5 * logf_target(x_proposal) - 0.5 * logf_proposal(x_proposal)).mean()
    return (target_term + proposal_term).item()



def nce_estim(x_target, x_proposal, logf_target, logf_proposal, loss="nce", verbose=True):
    """Compute the minimizer of the classification loss.
    Note: as it stands, x_target and x_proposal are torch tensors.
    logf_target and logf_proposal take as input torch tensors and spit out numpy arrays.
    """
    # choose the loss function
    if loss == "nce":
        loss_func = logistic_loss
        grad_func = grad_logistic_loss
    elif loss == "is":
        loss_func = is_loss
        grad_func = grad_is_loss
    elif loss == "rev-is":
        loss_func = revis_loss
        grad_func = grad_revis_loss
    elif loss == "exp":
        loss_func = exp_loss
        grad_func = grad_exp_loss

    # functions for the optimization
    fun = partial(
        loss_func,
        x_target=x_target, x_proposal=x_proposal, logf_target=logf_target, logf_proposal=logf_proposal)
    jac = partial(
        grad_func,
        x_target=x_target, x_proposal=x_proposal, logf_target=logf_target, logf_proposal=logf_proposal)
    callback = partial(
        callback_scipy_minimize,
        x_target=x_target, x_proposal=x_proposal, logf_target=logf_target, logf_proposal=logf_proposal,
        loss_func=loss_func, grad_func=grad_func,
        verbose=verbose)

    # run the optimization or directly get the result
    if loss == "nce":
        x0 = np.random.randn()
        beta_hat = minimize(
            fun, x0, args=(), method="CG", jac=jac,
            # tol=1e-20,
            callback=callback, options={"disp": verbose, "maxiter": None},
        )["x"].item()
    elif loss == "is":
        beta_hat = np.log(np.exp(logf_target(x_proposal) - logf_proposal(x_proposal)).mean())
    elif loss == "rev-is":
        beta_hat = -np.log(np.exp(logf_proposal(x_target) - logf_target(x_target)).mean())
    elif loss == "exp":
        beta_is = np.log(np.exp(logf_target(x_proposal) - logf_proposal(x_proposal)).mean())
        beta_revis = -np.log(np.exp(logf_proposal(x_target) - logf_target(x_target)).mean())
        beta_hat = beta_is + beta_revis
    return beta_hat


def annealed_nce_estim(
        n_distributions=2,
        sample_size=1000,
        n_repeats=2,
        n_jobs=3,
        proposal=MultivariateNormal(loc=torch.zeros(2), covariance_matrix=torch.eye(2)),
        target=MultivariateNormal(loc=torch.ones(2), covariance_matrix=torch.eye(2)),
        target_logZ=0.,
        target_logZ_estim=0.,
        path_name="arithmetic",
        loss="nce",
        verbose=1,
        two_step=False,
        two_step_path_name="geometric",
        two_step_n_distributions=10,
        two_step_n_avg=5,
        two_step_loss="is"
):
    all_target_logZ_preestim = [target_logZ_estim for _ in range(n_repeats)]

    if two_step:
        # Pre-Estimation
        two_step_n_distributions = 10

        # Define schedule
        ts = np.linspace(0, 1, two_step_n_distributions)

        # Define path
        path = UnnormalizedDistributionPath(
            target=target, proposal=proposal, path_name=two_step_path_name,
            target_logZ=target_logZ, target_logZ_estim=target_logZ)

        # Compute optimal MSE
        if (proposal.__class__ == MultivariateNormal) and (target.__class__ == MultivariateNormal):
            mse_best = path.compute_optimal_pathlength() / sample_size
            if loss == "exp":
                mse_best *= 2
        else:
            mse_best = None

        # Define sample shape for each classification task along the path
        sample_size_subtask = int(sample_size / 2 / (two_step_n_distributions - 1))

        # list of estimators, shape (n_repeats, n_distributions - 1)
        estimators = Parallel(n_jobs=n_jobs, verbose=verbose, backend='loky')(
            delayed(nce_estim)(
                x_target=path.sample(sample_shape=(sample_size_subtask,), t=ts[idx_t + 1]),
                x_proposal=path.sample(sample_shape=(sample_size_subtask,), t=ts[idx_t]),
                logf_target=partial(path.logf, t=ts[idx_t + 1]),
                logf_proposal=partial(path.logf, t=ts[idx_t]),
                loss=two_step_loss,
                verbose=verbose
            )
            for repeat, idx_t in product(range(n_repeats * two_step_n_avg), range(two_step_n_distributions - 1))
        )
        estimators = np.array(estimators).reshape(n_repeats * two_step_n_avg, two_step_n_distributions - 1)

        all_target_logZ_preestim = estimators.sum(axis=-1).reshape(n_repeats, two_step_n_avg).mean(axis=-1)  # (n_repeats,)

    # Estimation

    # Define schedule
    ts = np.linspace(0, 1, n_distributions)

    # Define path
    all_paths = [
        UnnormalizedDistributionPath(
            target=target, proposal=proposal, path_name=path_name,
            target_logZ=target_logZ, target_logZ_estim=_
        )
        for _ in all_target_logZ_preestim
    ]

    # Compute optimal MSE
    path = all_paths[0]
    if (proposal.__class__ == MultivariateNormal) and (target.__class__ == MultivariateNormal):
        mse_best = path.compute_optimal_pathlength() / sample_size
        if loss == "exp":
            mse_best *= 2
    else:
        mse_best = None

    # Define sample shape for each classification task along the path
    sample_size_subtask = int(sample_size / 2 / (n_distributions - 1))

    # list of estimators, shape (n_repeats, n_distributions - 1)
    estimators = Parallel(n_jobs=n_jobs, verbose=verbose, backend='loky')(
        delayed(nce_estim)(
            x_target=path.sample(sample_shape=(sample_size_subtask,), t=ts[idx_t + 1]),
            x_proposal=path.sample(sample_shape=(sample_size_subtask,), t=ts[idx_t]),
            logf_target=partial(path.logf, t=ts[idx_t + 1]),
            logf_proposal=partial(path.logf, t=ts[idx_t]),
            loss=loss,
            verbose=verbose
        )
        for (repeat, path), idx_t in product(zip(range(n_repeats), all_paths), range(n_distributions - 1))
    )
    estimators = np.array(estimators).reshape(n_repeats, n_distributions - 1)

    beta_hats = estimators.sum(axis=-1)   # (n_repeats,)

    return beta_hats, np.var(beta_hats), mse_best
