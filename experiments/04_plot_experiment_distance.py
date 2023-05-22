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

FONT = 50
plt.rcParams.update({
    "figure.titlesize": FONT,
    "legend.fontsize": FONT,
    "font.size": FONT,
    "axes.titlesize": FONT,
    "axes.labelsize": FONT}
)
# %%

# load data
EXPE_NAME = "param_distance"
results = torch.load(RESULTS_FOLDER / f"annealed_nce_expe_{EXPE_NAME}.th")
df = pd.DataFrame(results)
NDISTS = 10

# reformat dataframe
df = df.replace(
    to_replace="arithmetic-adaptive", value="two-step")
df = df.replace(
    to_replace="arithmetic-adaptive-trig", value="two-step (trig)")

# figure
fig, ax = plt.subplots(figsize=(9, 7))

# no path
sel = (df.path_name == "two-step (trig)") & (df.n_distributions == 2)
label = "no path"
ax.plot(
    df.loc[sel, "param_distance"], np.log10(df.loc[sel, "error"]),
    color="black", lw=10, label=label)
ax.scatter(
    df.loc[sel, "param_distance"], np.log10(df.loc[sel, "error"]),
    color="black", s=300, label=None)

# optimal path
label = "optimal"

ax.plot(
    df.loc[sel, "param_distance"], np.log10(df.loc[sel, "error_best"]),
    color="black", label=label,
    lw=20, ls="--",
    # marker="x", markersize=35, markerfacecolor="white", markeredgewidth=5, markevery=2
)


# different paths
path_names = [
    "geometric", "arithmetic", "two-step", "two-step (trig)"
]
colors = ["green", "red", "blue", "purple"]

for path_name, color in zip(path_names, colors):
    sel = (df.path_name == path_name) & (df.n_distributions == 10)

    label = path_name

    # path error
    ax.plot(
        df.loc[sel, "param_distance"], np.log10(df.loc[sel, "error"]),
        color=color, lw=10, label=label)
    ax.scatter(
        df.loc[sel, "param_distance"], np.log10(df.loc[sel, "error"]),
        color=color, s=300, label=None)


ax.set(
    ylabel="MSE (log10)",
    xlabel="Parameter distance",
    xlim=(0, 30),
    ylim=(-5.5, 0.),
    yticks=[-5, -2.5, 0.],
)
ax.spines[['right', 'top']].set_visible(False)

ldg = ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5), ncol=1, frameon=False)

fig.tight_layout()
plt.savefig(
    IMAGE_FOLDER / f"annealed_nce_{EXPE_NAME}.pdf",
    bbox_extra_artists=(ldg,), bbox_inches='tight'
)
# %%
