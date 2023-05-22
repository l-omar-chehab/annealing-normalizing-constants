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
EXPE_NAME = "unnormalization_2"
results = torch.load(RESULTS_FOLDER / f"annealed_nce_expe_{EXPE_NAME}.th")
df = pd.DataFrame(results)

INPUT = "target_logZ"
# HUE = "mean_norm"

path_names = ["geometric", "arithmetic", "arithmetic-adaptive-trig"]
cmaps = [cm.get_cmap('Greens', 4), cm.get_cmap('Reds', 4), cm.get_cmap('Blues', 4)]

# normalize = interp1d(
#     [df[HUE].unique().min(), df[HUE].unique().max()],
#     [0.2, 1]
# )
fig, ax = plt.subplots(figsize=(5, 3.5))

for path_name, cmap in zip(path_names, cmaps):
    # for hue in df[HUE].unique():
    #     sel = (df.path_name == path_name) & (df[HUE] == hue)
    #     color = cmap(normalize(hue))
    #     ax.scatter(
    #         x=df.loc[sel, INPUT],
    #         y=np.log10(df.loc[sel, "error"]),
    #         color=color,
    #         s=4,
    #     )
    #     ax.plot(
    #         df.loc[sel, INPUT],
    #         np.log10(df.loc[sel, "error"]),
    #         color=color,
    #         lw=3,
    #     )
    sel = (df.path_name == path_name)
    color = cmap(1)
    ax.scatter(
        x=df.loc[sel, INPUT],
        y=np.log10(df.loc[sel, "error"]),
        color=color,
        s=200,
    )
    ax.plot(
        df.loc[sel, INPUT],
        np.log10(df.loc[sel, "error"]),
        color=color,
        lw=6,
        label=path_name
    )

error_best = df.iloc[0].error_best
ax.axhline(y=np.log10(error_best), color="k", ls="--", lw=5, label="best")

ax.set(
    ylabel="MSE (log10)",
    xlabel=INPUT,
    xlim=(-4, 4)
    # ylim=(-6, -2)
)
ax.spines[['right', 'top']].set_visible(False)

ldg = ax.legend(
    bbox_to_anchor=(2.3, 0.8), ncol=1)


# ax.legend()
# norm = plt.Normalize(
#     df[HUE].unique().min(), df[HUE].unique().max())
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = ax.figure.colorbar(sm)
# cbar.ax.get_yaxis().labelpad = 30
# # cbar.ax.get_yaxis().set_ticks([2, 5, 9])
# cbar.ax.set_ylabel(HUE, rotation=270)

# fig.tight_layout()
plt.savefig(
    IMAGE_FOLDER / f"annealed_nce_{EXPE_NAME}.pdf",
    bbox_extra_artists=(ldg,), bbox_inches='tight'
)

# %%
