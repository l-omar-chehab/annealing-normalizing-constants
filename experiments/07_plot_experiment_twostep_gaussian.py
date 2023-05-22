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
EXPE_NAME = "twostep_gaussian"
results = torch.load(RESULTS_FOLDER / f"annealed_nce_expe_{EXPE_NAME}.th")
df = pd.DataFrame(results)

# sub-select and reformat
truth = df.iloc[0].target_logZ
estimates = np.concatenate(
    [df.iloc[row].logZs for row in range(len(df))]
).flatten()
log10errors = [np.log10(df.iloc[row].logZs.var()) for row in range(len(df))]
print("Errors (log10): ", log10errors)
path_names = np.concatenate([[df.iloc[row].path_name] * len(df.iloc[row].logZs) for row in range(len(df))]).flatten()
df = pd.DataFrame({"estimates": estimates, "path_names": path_names})

# figure
fig, ax = plt.subplots(figsize=(7.5, 2.7))

# sns.stripplot(
#     x="estimates", y="models", data=df,
#     jitter=0.1, ax=ax,
#     size=2.5,
#     order=["is", "rev-is", "exp", "nce"]
# )
sns.violinplot(
    x="estimates", y="path_names", data=df,  # hue="variance",
    width=0.5, linewidth=0.5,
    boxprops=dict(alpha=.15),
    ax=ax,
    # order=["is", "exp", "rev-is", "nce"],
    inner=None,
    # showmeans=True,
    # meanprops={"marker": "o",
    #            "markerfacecolor": "white",
    #            "markeredgecolor": "black",
    #            "markersize": "10"}
)

sns.despine(ax=ax)

ax.axvline(x=truth, color="black", linestyle="--", linewidth=0.5)

ax.set(
    ylabel="",
    xlabel="Estimates",
    # yticklabels=["IS", "IS-RevIS", "RevIS", "NCE"],
    # xtickslabels=[-0.5, 0, 0.5]
    # xlim=(truth + 0.2, truth - 0.2)
)
ax.set_ylabel("")

# ax.get_legend().remove()

fig.tight_layout()
plt.savefig(
    IMAGE_FOLDER / f"annealed_nce_{EXPE_NAME}.pdf",
)


# %%
