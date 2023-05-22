"""Plot experiments on Annealed NCE."""

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import torch
from scipy.interpolate import interp1d

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
# EXPE_NAME = "mean"
# EXPE_NAME = "fat_cov"
# EXPE_NAME = "thin_cov"
# EXPE_NAME = "thin_tail"
# EXPE_NAME = "fat_tail"
# EXPE_NAME = "mean_norm"
# EXPE_NAME = "cov_diag"
# EXPE_NAME = "meannorm_and_lognormalization_effects"
EXPE_NAME = "parameter_distance_and_normalization"
# EXPE_NAME = "gaussian_target"
# EXPE_NAME = "ica_target"


# PATH_NAME = "geometric"
# PATH_NAME = "convolutional"
PATH_NAME = "arithmetic"
# PATH_NAME = "arithmetic-fastschedule"
# PATH_NAME = "arithmetic-slowschedule"
# PATH_NAME = "arithmetic-normalized"
# PATH_NAME = "arithmetic-optimal-normalized"


# %%
results = torch.load(RESULTS_FOLDER / f"annealed_nce_expe_{EXPE_NAME}.th")
df = pd.DataFrame(results)


# %%
# Utils
def compute_mad(x):
    return np.median(np.abs(x - np.median(x)))



# %%
# Effect of annealing

cmap = cm.get_cmap('Reds', 12)
normalize = interp1d(
    [df.dim.unique().min(), df.dim.unique().max()],
    # [df.target_logZ.unique().min(), df.target_logZ.unique().max()],
    [0.2, 1]
)

fig, ax = plt.subplots(figsize=(7.5, 3.5))

# for target_logZ in df.target_logZ.unique():
for dim in df.dim.unique():
    # sel = (df.path_name == PATH_NAME) & (df.target_logZ == target_logZ)
    sel = (df.path_name == PATH_NAME) & (df.dim == dim)
    color = cmap(normalize(dim))
    ax.scatter(
        x=df.loc[sel, "n_distributions"],
        y=np.log10(df.loc[sel, "error"]),
        # y=df.loc[sel, "error"],
        color=color,
        s=4,
    )
    ax.plot(
        df.loc[sel, "n_distributions"],
        np.log10(df.loc[sel, "error"]),
        # df.loc[sel, "error"],
        color=color,
        lw=3,
    )

ax.set(
    ylabel="MSE (log10)",
    # ylabel="MSE",
    xlabel="Number of distributions",
    # ylim=(-6, -2)
)
ax.spines[['right', 'top']].set_visible(False)

# ax.legend(bbox_to_anchor=(1.1, 1.05))
norm = plt.Normalize(
    # df.target_logZ.unique().min(), df.target_logZ.unique().max())
    df.dim.unique().min(), df.dim.unique().max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = ax.figure.colorbar(sm)
cbar.ax.get_yaxis().labelpad = 30
cbar.ax.get_yaxis().set_ticks([2, 50, 100])
# cbar.ax.set_ylabel('Target logZ', rotation=270)
cbar.ax.set_ylabel('Dimension', rotation=270)

fig.tight_layout()
plt.savefig(IMAGE_FOLDER / f"mse_vs_ndistributions_expe_{EXPE_NAME}_path_{PATH_NAME}.pdf")

# %%


# %%
# Effect of mean difference

DIM = 2
# NDISTS = 3
PATH_DIFF = "mean_norm"  # "cov_diag"

# cmap = cm.get_cmap('Reds', 12)
normalize = interp1d(
    [df.n_distributions.unique().min(), df.n_distributions.unique().max()],
    [0.2, 1]
)
path_names = ["geometric"]   # "arithmetic"]
path_cmaps = [cm.get_cmap('Reds', 12), cm.get_cmap('Blues', 12)]


fig, ax = plt.subplots(figsize=(7.5, 3.5))

for path_name, path_cmap in zip(path_names, path_cmaps):
    for n_distributions in df.n_distributions.unique():
        sel = (df.path_name == path_name) & (df.dim == DIM) & (df.n_distributions == n_distributions)
        color = path_cmap(normalize(n_distributions))
        ax.scatter(
            x=df.loc[sel, PATH_DIFF],
            y=np.log10(df.loc[sel, "error"]),
            # y=df.loc[sel, "error"],
            color=color,
            label=path_name,
            s=50,
        )
        ax.plot(
            df.loc[sel, PATH_DIFF],
            np.log10(df.loc[sel, "error"]),
            # df.loc[sel, "error"],
            color=color,
            # label=path_name,
            lw=3,
        )

ax.set(
    ylabel="MSE (log10)",
    # ylabel="MSE",
    xlabel="Mean norm",
    # ylim=(-6, -2)
)
ax.spines[['right', 'top']].set_visible(False)

# ax.legend(bbox_to_anchor=(0.27, -0.45), ncol=3)
# ax.legend()

# norm = plt.Normalize(
#     df.dim.unique().min(), df.dim.unique().max())
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = ax.figure.colorbar(sm)
# cbar.ax.get_yaxis().labelpad = 30
# cbar.ax.get_yaxis().set_ticks([2, 50, 100])
# cbar.ax.set_ylabel('Dimension', rotation=270)

fig.tight_layout()
plt.savefig(IMAGE_FOLDER / f"mse_vs_density_gap_{PATH_DIFF}_nb_distributions_{NDISTS}.pdf")


# %%
# Effect of unnormalization

DIM = 2
NDISTS = 9
DISCRIM_1 = "target_logZ"
DISCRIM_2 = "mean_norm"

path_names = ["geometric", "arithmetic"]
cmaps = [cm.get_cmap('Reds', 12), cm.get_cmap('Blues', 12)]


normalize = interp1d(
    [0, len(df[DISCRIM_2].path_name.unique()) - 1],
    [0.2, 1]
)

fig, ax = plt.subplots(figsize=(7.5, 3.5))

for idx, (path_name, cmap) in enumerate(zip(path_names, cmaps)):
    sel = (df.path_name == path_name) & (df.dim == DIM) & (df.n_distributions == NDISTS)
    color = cmap(normalize(idx))
    ax.scatter(
        x=df.loc[sel, PATH_DIFF],
        y=np.log10(df.loc[sel, "error"]),
        # y=df.loc[sel, "error"],
        # color=color,
        label=path_name,
        s=50,
    )
    ax.plot(
        df.loc[sel, PATH_DIFF],
        np.log10(df.loc[sel, "error"]),
        # df.loc[sel, "error"],
        # color=color,
        # label=path_name,
        lw=3,
    )

ax.set(
    ylabel="MSE (log10)",
    # ylabel="MSE",
    xlabel="Mean norm",
    # ylim=(-6, -2)
)
ax.spines[['right', 'top']].set_visible(False)

# ax.legend(bbox_to_anchor=(0.27, -0.45), ncol=3)
ax.legend()

# norm = plt.Normalize(
#     df.dim.unique().min(), df.dim.unique().max())
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = ax.figure.colorbar(sm)
# cbar.ax.get_yaxis().labelpad = 30
# cbar.ax.get_yaxis().set_ticks([2, 50, 100])
# cbar.ax.set_ylabel('Dimension', rotation=270)

fig.tight_layout()
plt.savefig(IMAGE_FOLDER / f"mse_vs_density_gap_{PATH_DIFF}_nb_distributions_{NDISTS}.pdf")

# %%
# Joint Effect of parameter and unnormalization

DIM = 2
NDISTS = 9
INPUT = "target_logZ"
HUE = "mean_norm"

path_names = ["geometric", "arithmetic"]
cmaps = [cm.get_cmap('Reds', 12), cm.get_cmap('Blues', 12)]

normalize = interp1d(
    [df[HUE].unique().min(), df[HUE].unique().max()],
    [0.2, 1]
)
fig, ax = plt.subplots(figsize=(7.5, 3.5))

for path_name, cmap in zip(path_names, cmaps):
    for hue in df[HUE].unique():
        sel = (df.path_name == path_name) & (df.n_distributions == NDISTS) & (df[HUE] == hue)
        color = cmap(normalize(hue))
        ax.scatter(
            x=df.loc[sel, INPUT],
            y=np.log10(df.loc[sel, "error"]),
            # y=df.loc[sel, "error"],
            color=color,
            s=4,
        )
        ax.plot(
            df.loc[sel, INPUT],
            np.log10(df.loc[sel, "error"]),
            # df.loc[sel, "error"],
            color=color,
            lw=3,
        )

ax.set(
    ylabel="MSE (log10)",
    # ylabel="MSE",
    xlabel=INPUT,
    # ylim=(-6, -2)
)
ax.spines[['right', 'top']].set_visible(False)

norm = plt.Normalize(
    df[HUE].unique().min(), df[HUE].unique().max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = ax.figure.colorbar(sm)
cbar.ax.get_yaxis().labelpad = 30
# cbar.ax.get_yaxis().set_ticks([2, 5, 9])
cbar.ax.set_ylabel(HUE, rotation=270)

fig.tight_layout()
plt.savefig(IMAGE_FOLDER / f"mse_vs_param_and_lognorm_nb_distributions_{NDISTS}.pdf")


# %%
# ICA figure

# load data
EXPE_NAME = "ica_target"
results = torch.load(RESULTS_FOLDER / f"annealed_nce_expe_{EXPE_NAME}.th")
df = pd.DataFrame(results)

# hyperparameters
DIM = 50
IS_NORMALIZED = False

# color maps
normalize = interp1d(
    [df.n_distributions.unique().min(), df.n_distributions.unique().max()],
    [0.2, 1]
)
path_names = ["arithmetic", "arithmetic-adaptive"]
path_cmaps = [cm.get_cmap('Reds', 12), cm.get_cmap('Blues', 12)]
# path_names = path_names[:1]
# path_cmaps = path_cmaps[:1]


# figure
fig, ax = plt.subplots(figsize=(7.5, 3.5))

for path_name, path_cmap in zip(path_names, path_cmaps):
    for n_distributions in df.n_distributions.unique():
        sel_normalized = (df.target_logZ == 0) if IS_NORMALIZED else (df.target_logZ != 0)
        sel = (df.path_name == path_name) & (df.dim == DIM) & sel_normalized & (df.n_distributions == n_distributions)
        color = path_cmap(normalize(n_distributions))
        ax.plot(
            df.loc[sel, "dim"],
            np.log10(df.loc[sel, "error"]),
            # np.log10(
            #     [compute_mad(elem) for elem in df.loc[sel, "logZs"]]
            # ),
            # df.loc[sel, "error"],
            color=color,
            # label=path_name,
            lw=3,
        )

ax.set(
    ylabel="MSE (log10)",
    # ylabel="MSE",
    xlabel="Distance between the modes",
    # ylim=(-5, 0)
)
ax.spines[['right', 'top']].set_visible(False)

# ax.legend(bbox_to_anchor=(0.27, -0.45), ncol=3)
# ax.legend()

norm = plt.Normalize(
    df.n_distributions.unique().min(), df.n_distributions.unique().max())
sm = plt.cm.ScalarMappable(cmap=path_cmaps[0], norm=norm)
sm.set_array([])
cbar = ax.figure.colorbar(sm)
cbar.ax.get_yaxis().labelpad = 10
cbar.ax.get_yaxis().set_ticks([2, 10])
cbar.ax.set_ylabel('Nb Dists.', rotation=270)

fig.tight_layout()
plt.savefig(IMAGE_FOLDER / "annealed_nce_ica.pdf")




# %%
# ICA barplot

import seaborn as sns

# load data
EXPE_NAME = "ica_target"
results = torch.load(RESULTS_FOLDER / f"annealed_nce_expe_{EXPE_NAME}.th")
df = pd.DataFrame(results)

# sub-select and reformat
sel = (df.dim == 50) & (df.path_name == "arithmetic-schedule")
df = df[sel]
estimates = np.concatenate([df.iloc[row].logZs for row in range(len(df))]).flatten()
log10errors = [np.log10(df.iloc[row].logZs.var()) for row in range(len(df))]
print("Errors (log10): ", log10errors)
models = np.concatenate([[df.iloc[row].path_name + str(df.iloc[row].n_distributions)] * len(df.iloc[row].logZs) for row in range(len(df))]).flatten()
df = pd.DataFrame({"estimates": estimates, "models": models})

# figure
fig, ax = plt.subplots()

sns.stripplot(x="models", y="estimates", data=df,
              jitter=0.05, ax=ax)
sns.boxplot(x="models", y="estimates", data=df,
            width=0.5, linewidth=0.5, boxprops=dict(alpha=.15), ax=ax)
sns.despine(ax=ax)

# ax.set(
#     ylabel="Estimate",
#     xlabel="Loss",
#     xticklabels=["IS", "NCE"],
#     ylim=(50, 80)
# )
ax.set_ylabel("")

fig.tight_layout()
plt.savefig(IMAGE_FOLDER / "ica_target.pdf")


# %%
# Effect of loss

import seaborn as sns

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

# sns.stripplot(
#     x="estimates", y="models", data=df,
#     jitter=0.1, ax=ax,
#     size=2.5,
#     order=["is", "rev-is", "exp", "nce"]
# )
sns.violinplot(
    x="estimates", y="models", data=df,  # hue="variance",
    width=0.5, linewidth=0.5,
    boxprops=dict(alpha=.15),
    ax=ax,
    order=["is", "rev-is", "exp", "nce"],
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
    ylabel="Loss",
    xlabel="Estimates",
    yticklabels=["Is", "RevIS", "Exp", "NCE"],
    # ylim=(truth + 0.2, truth - 0.2)
)
ax.set_ylabel("")

# ax.get_legend().remove()

fig.tight_layout()
plt.savefig(IMAGE_FOLDER / "nce_losses_comparison.pdf")

# %%


# %%
# Effect of the dimension

# load data
EXPE_NAME = "dim_var_diff"  # dim_var_diff
results = torch.load(RESULTS_FOLDER / f"annealed_nce_expe_{EXPE_NAME}.th")
df = pd.DataFrame(results)

normalize = interp1d(
    [df.n_distributions.unique().min(), df.n_distributions.unique().max()],
    [0.2, 1]
)
path_names = ["geometric", "arithmetic-adaptive"]  # , "arithmetic-adaptive"]
path_cmaps = [cm.get_cmap('Greens', 12), cm.get_cmap('Reds', 12)]  # , cm.get_cmap('Blues', 12), cm.get_cmap('Purples', 12)]


fig, ax = plt.subplots(figsize=(7.5, 3.5))

for path_name, path_cmap in zip(path_names, path_cmaps):
    print(path_name)
    print(path_cmap)
    for n_distributions in df.n_distributions.unique():
        sel = (df.path_name == path_name) & (df.n_distributions == n_distributions)
        color = path_cmap(normalize(n_distributions))
        ax.scatter(
            x=df.loc[sel, "dim"],
            y=np.log10(df.loc[sel, "error"]),
            color=color,
            s=20,
        )
        ax.plot(
            df.loc[sel, "dim"],
            np.log10(df.loc[sel, "error"]),
            color=color,
            lw=3,
        )

ax.set(
    ylabel="MSE (log10)",
    xlabel="Dimension",
    xlim=(0, 50),
    ylim=(-5, 0)
)
ax.spines[['right', 'top']].set_visible(False)

norm = plt.Normalize(
    df.n_distributions.unique().min(), df.n_distributions.unique().max())
sm = plt.cm.ScalarMappable(cmap=path_cmap, norm=norm)
sm.set_array([])
cbar = ax.figure.colorbar(sm)
cbar.ax.get_yaxis().labelpad = 30
cbar.ax.get_yaxis().set_ticks([2, 10])
cbar.ax.set_ylabel('Nb distribs.', rotation=270)

fig.tight_layout()
plt.savefig(IMAGE_FOLDER / f"mse_vs_dim_expe_{EXPE_NAME}_path_{PATH_NAME}.pdf")
# %%


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
# Gaussian figure

# load data
EXPE_NAME = "param_distance_2"
results = torch.load(RESULTS_FOLDER / f"annealed_nce_expe_{EXPE_NAME}.th")
df = pd.DataFrame(results)

# hyperparameters
DIM = 50
IS_NORMALIZED = False

# color maps
normalize = interp1d(
    [df.n_distributions.unique().min(), df.n_distributions.unique().max()],
    [0.2, 1]
)
path_names = ["geometric", "arithmetic", "arithmetic-adaptive-trig"]
path_cmaps = [cm.get_cmap('Greens', 12), cm.get_cmap('Reds', 12), cm.get_cmap('Blues', 12)]
path_names = path_names[2:3]
path_cmaps = path_cmaps[2:3]

# figure
fig, ax = plt.subplots(figsize=(10, 2.5))

for path_name, path_cmap in zip(path_names, path_cmaps):
    for n_distributions in df.n_distributions.unique():
        sel_normalized = (df.target_logZ == 0) if IS_NORMALIZED else (df.target_logZ != 0)
        sel = (df.path_name == path_name) & (df.dim == DIM) & sel_normalized & (df.n_distributions == n_distributions)
        color = path_cmap(normalize(n_distributions))

        # path error
        label = path_name if n_distributions == df.n_distributions.unique()[-1] else None

        # ax.scatter(
        #     x=df.loc[sel, "param_distance"],
        #     y=np.log10(df.loc[sel, "error"]),
        #     color=color,
        #     label=None,
        #     s=50,
        # )

        ax.plot(
            df.loc[sel, "param_distance"],
            np.log10(df.loc[sel, "error"]),
            color=color,
            lw=4,
            label=label,
        )

        # optimal path error
        label = "Optimal" if (n_distributions == df.n_distributions.unique()[-1] and path_name == path_names[-1]) else None

        # ax.scatter(
        #     x=df.loc[sel, "param_distance"],
        #     y=np.log10(df.loc[sel, "error_best"]),
        #     color=color,
        #     label=None,
        #     s=50,
        # )

        ax.plot(
            df.loc[sel, "param_distance"],
            np.log10(df.loc[sel, "error_best"]),
            color="black",
            lw=4,
            ls='--',
            label=label
        )

ax.set(
    ylabel="MSE (log10)",
    xlabel="Parameter distance",
    xlim=(0, 30),
    # ylim=(-6, 0),
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
# load data
EXPE_NAME = "dim_var_diff"
results = torch.load(RESULTS_FOLDER / f"annealed_nce_expe_{EXPE_NAME}.th")
df = pd.DataFrame(results)
VAR = 1. / 4
NDISTS = 10

# reformat dataframe
df = df.replace(to_replace="arithmetic-adaptive", value="arithmetic (oracle)")
df = df.replace(to_replace="arithmetic-adaptive-trig", value="arithmetic (oracle-trig)")

# figure
fig, ax = plt.subplots(figsize=(10, 4))

# no path
sel = (df.path_name == "arithmetic (oracle-trig)") & (df.n_distributions == 2)
label = "no path"
ax.plot(
    df.loc[sel, "dim"], np.log10(df.loc[sel, "error"]),
    color="black", lw=6, label=label)

# different paths
path_names = [
    "geometric", "arithmetic", "arithmetic (oracle)", "arithmetic (oracle-trig)"
]
colors = ["green", "red", "blue", "purple"]

for path_name, color in zip(path_names, colors):
    sel_var = [df.target[idx].covariance_matrix.diag().unique().item() == VAR for idx in range(len(df))]
    sel = (df.path_name == path_name) & (df.n_distributions == 10) & sel_var

    label = path_name

    # path error
    ax.plot(
        df.loc[sel, "dim"], np.log10(df.loc[sel, "error"]),
        color=color, lw=6, label=label)

    # optimal path error
    label = "Optimal" if (path_name == path_names[0]) else None

    ax.plot(
        df.loc[sel, "dim"], np.log10(df.loc[sel, "error_best"]),
        color="black", lw=6, ls='--', label=label)

ax.set(
    ylabel="MSE (log10)",
    xlabel="Dimension",
    xlim=(0, 50),
    ylim=(-5.1, -0),
    yticks=[-5, -2.5, 0],
)
ax.spines[['right', 'top']].set_visible(False)

ldg = ax.legend(bbox_to_anchor=(1.05, -0.35), ncol=2, frameon=False)

# fig.tight_layout()
# plt.savefig(
#     IMAGE_FOLDER / f"annealed_nce_{EXPE_NAME}.pdf",
#     bbox_extra_artists=(ldg,), bbox_inches='tight'
# )
# %%