"""Plot experiments on Annealed NCE."""

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import torch
from scipy.interpolate import interp1d
import seaborn as sns

from annealednce.defaults import RESULTS_FOLDER, IMAGE_FOLDER


plt.style.use("/Users/omar/Downloads/matplotlib_style.txt")

plt.rcParams.update({
    "figure.titlesize": 22,
    "legend.fontsize": 22,
    "font.size": 22,
    "axes.titlesize": 22,
    "axes.labelsize": 22}
)

# %%

# load data
EXPE_NAME = "dim_mean_diff"
results = torch.load(RESULTS_FOLDER / f"annealed_nce_expe_{EXPE_NAME}.th")
df = pd.DataFrame(results)
df = df.replace(to_replace="arithmetic-adaptive", value="arithmetic (normalized)")

normalize = interp1d(
    [df.n_distributions.unique().min(), df.n_distributions.unique().max()],
    [0.2, 1]
)
path_names = ["arithmetic", "arithmetic (normalized)", "geometric"]
path_cmaps = [cm.get_cmap('Reds', 12), cm.get_cmap('Blues', 12), cm.get_cmap('Greens', 12)]

fig, ax = plt.subplots(figsize=(10, 2.5))

for path_name, path_cmap in zip(path_names, path_cmaps):
    for n_distributions in df.n_distributions.unique():
        sel = (df.path_name == path_name) & (df.n_distributions == n_distributions)
        color = path_cmap(normalize(n_distributions))

        # path error
        label = path_name if n_distributions == df.n_distributions.unique()[-1] else None

        # ax.scatter(
        #     x=df.loc[sel, "dim"],
        #     y=np.log10(df.loc[sel, "error"]),
        #     color=color,
        #     label=None,
        #     s=50,
        # )

        ax.plot(
            df.loc[sel, "dim"],
            np.log10(df.loc[sel, "error"]),
            color=color,
            lw=4,
            label=label,
        )

        # optimal path error
        label = "Optimal" if (n_distributions == df.n_distributions.unique()[-1] and path_name == path_names[-1]) else None

        # ax.scatter(
        #     x=df.loc[sel, "dim"],
        #     y=np.log10(df.loc[sel, "error_best"]),
        #     color=color,
        #     label=None,
        #     s=50,
        # )

        ax.plot(
            df.loc[sel, "dim"],
            np.log10(df.loc[sel, "error_best"]),
            color="black",
            lw=3,
            ls='--',
            label=label
        )

ax.set(
    ylabel="MSE (log10)",
    xlabel="Dimension",
    xlim=(0, 50),
    ylim=(-5.4, 0)
)
ax.spines[['right', 'top']].set_visible(False)

ldg = ax.legend(bbox_to_anchor=(1.2, -0.35), ncol=2)

norm = plt.Normalize(
    df.n_distributions.unique().min(), df.n_distributions.unique().max())
sm = plt.cm.ScalarMappable(cmap=path_cmaps[0], norm=norm)
sm.set_array([])
cbar = ax.figure.colorbar(sm)
cbar.ax.get_yaxis().labelpad = 10
cbar.ax.get_yaxis().set_ticks([2, 10])
cbar.ax.set_ylabel('Nb Dists.', rotation=270)

# fig.tight_layout()
plt.savefig(
    IMAGE_FOLDER / f"annealed_nce_{EXPE_NAME}.pdf",
    bbox_extra_artists=(ldg,), bbox_inches='tight'
)
# %%
