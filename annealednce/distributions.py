"""Class for distribution path."""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal
from scipy.special import logsumexp


# Utils
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Distribution paths
class UnnormalizedDistributionPath(nn.Module):
    """Path of distributions, defined in the space of unnormalized densities."""
    def __init__(self,
                 proposal, target,
                 proposal_logZ=0., target_logZ=0., target_logZ_estim=0.,
                 path_name="arithmetic"):
        super().__init__()
        self.proposal = proposal
        self.target = target
        self.path_name = path_name
        self.proposal_logZ = proposal_logZ
        self.target_logZ = target_logZ
        self.target_logZ_estim = target_logZ_estim

    def forward(self, *args):
        raise RuntimeError("forward method is not implemented for a Path object.")

    # Define the path in the space of unnormalized densities f(x, t)
    def logf_proposal(self, x):
        """Log of the unnormalized proposal density."""
        return self.proposal.log_prob(x) + self.proposal_logZ

    def logf_target(self, x):
        """Log of the unnormalized target density."""
        return self.target.log_prob(x) + self.target_logZ

    def log_adaptive_weight_trig_unnormalized(self, t):
        """Adaptive weights for the arithmetic path in the space of unnormalized densities."""
        tau = np.sin(np.pi * t / 2)**2
        logw_t = np.log(tau) - \
            logsumexp(a=[np.log(1 - tau) + self.target_logZ_estim, np.log(tau)])
        return logw_t

    def log_adaptive_weight_unnormalized(self, t):
        """Adaptive weights for the arithmetic path in the space of unnormalized densities."""
        tau = t
        logw_t = np.log(tau) - \
            logsumexp(a=[np.log(1 - tau) + self.target_logZ_estim, np.log(tau)])
        return logw_t

    def log_adaptive_weight_trig_normalized(self, t):
        """Adaptive weights for the arithmetic path in the space of normalized densities."""
        tau = np.sin(np.pi * t / 2)**2
        logw_t = np.log(tau) - \
            logsumexp(a=[np.log(1 - tau) + self.target_logZ_estim - self.target_logZ, np.log(tau)])
        return logw_t

    def log_adaptive_weight_normalized(self, t):
        """Adaptive weights for the arithmetic path in the space of normalized densities."""
        tau = t
        logw_t = np.log(tau) - \
            logsumexp(a=[np.log(1 - tau) + self.target_logZ_estim - self.target_logZ, np.log(tau)])
        return logw_t

    def logf(self, x, t):
        """Log of the unnormalized path density.

        Parameters
        ----------
        x : torch tensor of shape (*sample_shape, n_comp)

        Returns
        -------
        logf_t : torch.tensor of shape (*sample_shape)
        """
        if self.path_name == "geometric":
            logf_t = (1 - t) * self.logf_proposal(x) + t * self.logf_target(x)
            return logf_t.numpy()
        elif self.path_name == "arithmetic":
            a = np.stack([self.logf_proposal(x), self.logf_target(x)], axis=-1)  # (sample_size, 2)
            b = np.array([1 - t, t])[None, :]                                    # (1, 2)
            logf_t = logsumexp(a=a, b=b, axis=-1)
            return logf_t
        elif self.path_name == "arithmetic-adaptive-trig":
            # compute mixture weights
            w_t = np.exp(self.log_adaptive_weight_trig_unnormalized(t))
            # compute mixture of unnormalized pdfs (in log space)
            a = np.stack([self.proposal.log_prob(x), self.target.log_prob(x)], axis=-1)  # (sample_size, 2)
            b = np.array([1 - w_t, w_t])[None, :]                                        # (1, 2)
            logf_t = logsumexp(a=a, b=b, axis=-1)
            return logf_t
        elif self.path_name == "arithmetic-adaptive":
            # compute mixture weights
            w_t = np.exp(self.log_adaptive_weight_unnormalized(t))
            # compute mixture of unnormalized pdfs (in log space)
            a = np.stack([self.proposal.log_prob(x), self.target.log_prob(x)], axis=-1)  # (sample_size, 2)
            b = np.array([1 - w_t, w_t])[None, :]                                        # (1, 2)
            logf_t = logsumexp(a=a, b=b, axis=-1)
            return logf_t
        elif self.path_name == "arithmetic-schedule":
            a = np.stack([self.logf_proposal(x), self.logf_target(x)], axis=-1)  # (sample_size, 2)
            power = np.log(1 + np.exp(self.target_logZ)) / np.log(2)
            b = np.array([1 - t**power, t**power])[None, :]                                    # (1, 2)
            logf_t = logsumexp(a=a, b=b, axis=-1)
            return logf_t

    # Sample the path
    def sample(self, t, sample_shape=()):
        """Samples x from pdf(x).

        Parameters
        ----------
        sample_shape : tuple of iid realizations, e.g. (n_samples,)

        Returns
        -------
        samples : torch.tensor of shape (*sample_shape, n_comp)
                  Samples from the pdf.
        """
        x_proposal = self.proposal.sample(sample_shape)
        x_target = self.target.sample(sample_shape)                   # (*sample_shape, dim)
        # dim = x_proposal.shape[-1]
        if self.path_name == "geometric":
            if (self.proposal.__class__ == MultivariateNormal) and (self.target.__class__ == MultivariateNormal):
                precision_matrix = (1 - t) * self.proposal.precision_matrix + t * self.target.precision_matrix
                loc = (1 - t) * self.proposal.precision_matrix @ self.proposal.loc + t * self.target.precision_matrix @ self.target.loc
                loc = torch.linalg.solve(precision_matrix, loc)
                dist = MultivariateNormal(loc=loc, precision_matrix=precision_matrix)
                x_t = dist.sample(sample_shape)
                return x_t
            else:
                raise NotImplementedError("For a generic statistical model, it is not clear \
                    what transformation in sample space, corresponds to a geometric path in distribution space.")
        elif self.path_name == "arithmetic":
            # decompose the log for numerical stability
            w_t = sigmoid(np.log(t) - np.log(1 - t) + self.target_logZ - self.proposal_logZ)
            # (1 - t) * self.proposal.log_prob(x_proposal) + t * self.target.log_prob(x_target)
            z_t = torch.bernoulli(w_t * torch.ones(*sample_shape, 1))   # (*sample_shape, 1)
            x_t = z_t * x_target + (1 - z_t) * x_proposal
            return x_t
        elif self.path_name == "arithmetic-adaptive-trig":
            # compute mixture weights
            w_t = np.exp(self.log_adaptive_weight_trig_normalized(t))
            # sample mixture of normalized pdfs
            z_t = torch.bernoulli(w_t * torch.ones(*sample_shape, 1))   # (*sample_shape, 1)
            x_t = z_t * x_target + (1 - z_t) * x_proposal
            return x_t
        elif self.path_name == "arithmetic-adaptive":
            # compute mixture weights
            w_t = np.exp(self.log_adaptive_weight_normalized(t))
            # sample mixture of normalized pdfs
            z_t = torch.bernoulli(w_t * torch.ones(*sample_shape, 1))   # (*sample_shape, 1)
            x_t = z_t * x_target + (1 - z_t) * x_proposal
            return x_t
        elif self.path_name == "arithmetic-schedule":
            # decompose the log for numerical stability
            power = np.log(1 + np.exp(self.target_logZ)) / np.log(2)
            w_t = sigmoid(np.log(t**power) - np.log(1 - t**power) + self.target_logZ - self.proposal_logZ)
            z_t = torch.bernoulli(w_t * torch.ones(*sample_shape, 1))   # (*sample_shape, 1)
            x_t = z_t * x_target + (1 - z_t) * x_proposal
            return x_t

    def log_prob(self, x, t):
        """Computes logpdf(x).

        Parameters
        ----------
        x : torch tensor of shape (*sample_shape, n_comp)

        Returns
        -------
        logprob : torch.tensor of shape (*sample_shape)
        """
        if self.path_name == "geometric":
            if (self.proposal.__class__ == MultivariateNormal) and (self.target.__class__ == MultivariateNormal):
                precision_matrix = (1 - t) * self.proposal.precision_matrix + t * self.target.precision_matrix
                loc = (1 - t) * self.proposal.precision_matrix @ self.proposal.loc + t * self.target.precision_matrix @ self.target.loc
                loc = torch.linalg.solve(precision_matrix, loc)
                dist = MultivariateNormal(loc=loc, precision_matrix=precision_matrix)
            return dist.log_prob(x).numpy()
        elif self.path_name == "arithmetic":
            a = np.stack([self.proposal.log_prob(x), self.target.log_prob(x)], axis=-1)    # (sample_size, 2)
            b = np.array([1 - t, t])[None, :]                                              # (1, 2)
            logf_t = logsumexp(a=a, b=b, axis=-1)
            return logf_t
        if self.path_name == "optimal":
            if (self.proposal.__class__ == MultivariateNormal) and (self.target.__class__ == MultivariateNormal):
                det_num = torch.det(self.proposal.covariance_matrix)**(1 / 4) * torch.det(self.target.covariance_matrix)**(1 / 4)
                det_denom = torch.det(0.5 * self.proposal.covariance_matrix + 0.5 * self.target.covariance_matrix)**(1 / 2)
                exp_term = torch.exp((-1. / 8) * torch.dot(
                    self.target.loc - self.proposal.loc,
                    torch.linalg.solve(
                        0.5 * self.proposal.covariance_matrix + 0.5 * self.target.covariance_matrix,
                        self.target.loc - self.proposal.loc)
                ))
                hellinger_squared = 1 - det_num / det_denom * exp_term
                alpha_hellinger = torch.arctan(torch.sqrt(hellinger_squared / (4 - hellinger_squared)))
                coef_1 = np.cos((2 * t - 1) * alpha_hellinger) / (2 * torch.cos(alpha_hellinger))
                coef_2 = np.sin((2 * t - 1) * alpha_hellinger) / (2 * torch.sin(alpha_hellinger))
                a_t = coef_1 - coef_2
                b_t = coef_1 + coef_2
                logps = np.stack([0.5 * self.proposal.log_prob(x),
                                  0.5 * self.target.log_prob(x)], axis=-1)  # (sample_size, 2)
                weights = np.array([a_t, b_t])[None, :]                     # (1, 2)
                logf_t = 2 * logsumexp(a=logps, b=weights, axis=-1)
                return logf_t
            else:
                raise NotImplementedError("The optimal path is not implemented in the general case.")

    def compute_alpha_hellinger(self):
        if (self.proposal.__class__ == MultivariateNormal) and (self.target.__class__ == MultivariateNormal):
            det_num = torch.det(self.proposal.covariance_matrix)**(1 / 4) * torch.det(self.target.covariance_matrix)**(1 / 4)
            det_denom = torch.det(0.5 * self.proposal.covariance_matrix + 0.5 * self.target.covariance_matrix)**(1 / 2)
            exp_term = torch.exp((-1. / 8) * torch.dot(
                self.target.loc - self.proposal.loc,
                torch.linalg.solve(
                    0.5 * self.proposal.covariance_matrix + 0.5 * self.target.covariance_matrix,
                    self.target.loc - self.proposal.loc)
            ))
            hellinger_squared = 1 - det_num / det_denom * exp_term
            alpha_hellinger = torch.arctan(torch.sqrt(hellinger_squared / (2 - hellinger_squared)))
            return alpha_hellinger.item()
        else:
            raise NotImplementedError("The optimal path is not implemented in the general case.")

    def compute_optimal_pathlength(self):
        if (self.proposal.__class__ == MultivariateNormal) and (self.target.__class__ == MultivariateNormal):
            alpha_hellinger = self.compute_alpha_hellinger()
            return 16 * alpha_hellinger**2
        else:
            raise NotImplementedError("The optimal path length is not implemented in the general case.")
