"""Plot distributions along a path."""

# %%
import numpy as np
import torch
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt

from annealednce.distributions import UnnormalizedDistributionPath
from annealednce.defaults import IMAGE_FOLDER


plt.style.use("/Users/omar/Downloads/matplotlib_style.txt")
plt.rcParams.update({
    "figure.titlesize": 14,
    "legend.fontsize": 14,
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14}
)

# %%
# Hyperparameters
n_distributions = 3
path_names = [
    "geometric", "arithmetic",
    # , "convolutional",
    # "dimensional",
    "optimal"]
ts = np.linspace(0., 1., n_distributions)

# Statistical model
mean_proposal = torch.zeros(2)
covariance_proposal = torch.eye(2)

mean_target = torch.Tensor([5, 0])
covariance_target = torch.Tensor([[1, 0], [0, 5]])

proposal = MultivariateNormal(loc=mean_proposal, covariance_matrix=covariance_proposal)
target = MultivariateNormal(loc=mean_target, covariance_matrix=covariance_target)

# %%
# Plot
nrows, ncols = len(path_names), len(ts)
fig, axes = plt.subplots(
    # figsize=(1.5 * ncols, 1.5 * nrows),
    figsize=(7.5, 3.5),
    nrows=nrows, ncols=ncols)  # , sharex=True, sharey=True)
axes = axes.reshape(nrows, ncols)  # in case nrows=1, it does not remove the first axis

xx = yy = torch.linspace(-10., 10., 500)
XX, YY = torch.meshgrid(xx, yy)

for (row, path_name) in enumerate(path_names):
    for (col, t) in enumerate(ts):
        path = UnnormalizedDistributionPath(target=target, proposal=proposal, path_name=path_name)
        if path_name == "optimal":
            Z = path.log_prob(x=torch.stack([XX.flatten(), YY.flatten()], dim=-1), t=t)
            img = axes[row, col].contour(
                Z.reshape(500, 500).T, levels=30,
                colors="#1f77b4", linestyles="solid", linewidths=1.)
        else:
            x = path.sample(sample_shape=(10000,), t=t)
            axes[row, col].scatter(x[:, 0], x[:, 1], s=0.5, color='#1f77b4')

for (row, path_name) in enumerate(path_names):
    axes[row, 0].set_ylabel(path_name, rotation='horizontal', labelpad=50)
for (col, t) in enumerate(ts):
    axes[0, col].set(title=f"t = {t}")
for ax in axes[:-1].flatten():
    ax.set(xlim=(-8, 8), ylim=(-8, 8))
for ax in axes.flatten():
    ax.set(xticks=[], yticks=[])  # , xticks=[-5, 0, 5], yticks=[-5, 0, 5])
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)
#     ax.grid(False, linestyle='-.')

fig.subplots_adjust(wspace=0., hspace=0.)
fig.tight_layout()
# fig.savefig("/Users/omar/Downloads/paths.pdf")
fig.savefig(IMAGE_FOLDER / "paths.png")

# %%
