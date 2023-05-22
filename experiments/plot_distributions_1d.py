"""Plot distributions along a path."""

# %%
import numpy as np
import torch
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

from annealednce.distributions import UnnormalizedDistributionPath
from annealednce.defaults import RESULTS_FOLDER, IMAGE_FOLDER


plt.style.use("/Users/omar/Downloads/matplotlib_style.txt")

plt.rcParams.update({
    "figure.titlesize": 28,
    "legend.fontsize": 28,
    "font.size": 28,
    "axes.titlesize": 28,
    "axes.labelsize": 28}
)
# %%
# Hyperparameters
n_distributions = 5
path_names = [
    "geometric",
    # "convolutional",
    "arithmetic",
    # "dimensional",
    # "optimal"
]
ts = np.linspace(0., 1., n_distributions)

# Statistical model
mean_proposal = torch.zeros(1)
covariance_proposal = torch.eye(1)

mean_target = 10 * torch.ones(1)
covariance_target = torch.eye(1)

proposal = MultivariateNormal(loc=mean_proposal, covariance_matrix=covariance_proposal)
# target = MultivariateNormal(loc=mean_target, covariance_matrix=covariance_target)


class TargetMixture(torch.nn.Module):
    """."""
    def __init__(self):
        super().__init__()
        mix = torch.distributions.Categorical(torch.ones(2,))
        comp = torch.distributions.Normal(torch.Tensor([-7, 7]), (1 / np.sqrt(2)) * torch.ones(2))
        self.gmm = torch.distributions.MixtureSameFamily(mix, comp)

    def forward(self, *args):
        raise RuntimeError("forward method is not implemented for a Path object.")

    def log_prob(self, x):
        return self.gmm.log_prob(x[..., 0])

    def sample(self, sample_shape=()):
        return self.gmm.sample(sample_shape=sample_shape).reshape(sample_shape + (1,)) 


target = TargetMixture()



# %%
# Plot
nrows = len(path_names)
fig, axes = plt.subplots(
    figsize=(10, 3),
    nrows=nrows, ncols=1,
    sharex=True, sharey=True)
axes = axes.reshape(nrows, 1)  # in case nrows=1, it does not remove the first axis

cmap = cm.get_cmap('coolwarm')

xs = torch.linspace(-10, 10, 500).reshape(-1, 1)

for (row, path_name) in enumerate(path_names):
    for t in ts:
        color = cmap(1 - t)
        path = UnnormalizedDistributionPath(target=target, proposal=proposal, path_name=path_name)
        # ps = np.exp(path.log_prob(x=xs, t=t))
        ps = np.exp(path.logf(x=xs, t=t))
        ps = ps / ps.sum()
        axes[row, 0].plot(xs, ps, color=color, lw=5)

for (row, path_name) in enumerate(path_names):
    label = r"$q \rightarrow 0$" if path_name == "geometric" else r"$q = 1$"
    axes[row, 0].set_ylabel(label, rotation='horizontal', labelpad=50)
    axes[row, 0].yaxis.set_label_position("right")
    axes[row, 0].yaxis.tick_right()
for ax in axes[:-1].flatten():
    ax.set(xlim=(-10, 10))
    # ax.set(ylim=(-8, 8))
for ax in axes.flatten():
    ax.set(xticks=[], yticks=[])  # , xticks=[-5, 0, 5], yticks=[-5, 0, 5])
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)
#     ax.grid(False, linestyle='-.')

red_line = Line2D(
    [0], [0], label=r'$p_0$', color="red"
)
blue_line = Line2D(
    [0], [0], label=r'$p_1$', color="blue"
)

ldg = ax.legend(
    handles=[red_line, blue_line],
    ncol=2,
    loc='lower center', bbox_to_anchor=(0.5, -1),
    frameon=False,
)

# fig.subplots_adjust(wspace=0., hspace=0.7)
# fig.tight_layout()
fig.savefig(
    IMAGE_FOLDER / "annealed_nce_paths_1d.pdf",
    bbox_extra_artists=(ldg,), bbox_inches='tight'
)

# %%
