"""Plot experiments on Annealed NCE."""

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import seaborn as sns

from annealednce.defaults import RESULTS_FOLDER, IMAGE_FOLDER, ROOT_FOLDER


plt.style.use(ROOT_FOLDER / "matplotlib_style.txt")

plt.rcParams.update({
    "figure.titlesize": 22,
    "legend.fontsize": 22,
    "font.size": 22,
    "axes.titlesize": 22,
    "axes.labelsize": 22}
)

# %%

# load data
EXPE_NAME = "loss"
results = torch.load(RESULTS_FOLDER / f"annealed_nce_expe_{EXPE_NAME}.th")
df = pd.DataFrame(results)

# sub-select and reformat
sel = (df.dim == 50) & (df.n_distributions == 3) & (df.path_name == "geometric") & (df.variance == 2.)
df = df[sel]
truth = df.iloc[0].target_logZ
estimates = np.concatenate(
    [df.iloc[row].logZs for row in range(len(df))]
).flatten()
log10errors = [np.log10(df.iloc[row].logZs.var()) for row in range(len(df))]
print("Errors (log10): ", log10errors)
models = np.concatenate([[df.iloc[row].loss] * len(df.iloc[row].logZs) for row in range(len(df))]).flatten()
variances = np.concatenate([[df.iloc[row].variance] * len(df.iloc[row].logZs) for row in range(len(df))]).flatten()
df = pd.DataFrame({"estimates": estimates, "models": models, "variance": variances})

# figure
fig, ax = plt.subplots(figsize=(7.5, 2.7))

sns.violinplot(
    x="estimates", y="models", data=df,
    width=0.5, linewidth=0.5,
    boxprops=dict(alpha=.15),
    ax=ax,
    order=["is", "rev-is", "nce"],
    inner=None,
)

sns.despine(ax=ax)

ax.axvline(x=truth, color="black", linestyle="--", linewidth=0.5)

ax.set(
    ylabel="",
    xlabel="Estimates",
    yticklabels=["IS", "RevIS", "NCE"]
)
ax.set_ylabel("")

# ax.get_legend().remove()

fig.tight_layout()
plt.savefig(IMAGE_FOLDER / "nce_losses_comparison.pdf")

# %%
