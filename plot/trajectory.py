import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# def cell_clusters(
#     adata,
#     palette=None,
#     ax=None,
# ):
#


def trajectories(
    adata,
    trajectory_names=None,
    palette=None,
    ax=None,
):
    if trajectory_names is None:
        trajectory_names = adata.uns["decipher"]["trajectories"].keys()
    if type(trajectory_names) == str:
        trajectory_names = [trajectory_names]
    if ax is None:
        ax = plt.gca()

    default_color_palette = sns.color_palette(n_colors=len(trajectory_names))
    for i, t_name in enumerate(trajectory_names):
        cluster_locations = adata.uns["decipher"]["trajectories"][t_name]["cluster_locations"]
        if palette is not None and t_name in palette:
            color = palette[t_name]
        else:
            color = default_color_palette[i]
        ax.plot(
            cluster_locations[:, 0],
            cluster_locations[:, 1],
            marker="o",
            c="black",
            markerfacecolor=color,
            markersize=7,
        )
        ax.plot(
            cluster_locations[:1, 0],
            cluster_locations[:1, 1],
            marker="*",
            markersize=20,
            c="black",
            markerfacecolor=color,
        )


def gene_patterns(
    adata,
    gene_name,
    crop_to_min_length=False,
    smoothing_window=5,
    palette=None,
    label_palette=None,
    gene_patterns_names=None,
    figsize=[3, 2.3],
):
    if type(gene_name) == list:
        gene_id = [adata.var_names.tolist().index(gn) for gn in gene_name]
    else:
        gene_id = [adata.var_names.tolist().index(gene_name)]

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), "same") / np.convolve(np.ones_like(x), np.ones(w), "same")

    if gene_patterns_names is None:
        gene_patterns_names = list(adata.uns["decipher"]["gene_patterns"].keys())
    elif type(gene_patterns_names) == str:
        gene_patterns_names = [gene_patterns_names]

    fig = plt.figure(figsize=figsize)
    start_times = []
    end_times = []

    default_color_palette = sns.color_palette(n_colors=len(gene_patterns_names))
    for i, gp_name in enumerate(gene_patterns_names):
        gene_pattern = adata.uns["decipher"]["gene_patterns"][gp_name]

        gene_pattern_mean = moving_average(
            gene_pattern["mean"][:, gene_id].mean(axis=1), smoothing_window
        )
        gene_pattern_q25 = moving_average(
            gene_pattern["q25"][:, gene_id].mean(axis=1), smoothing_window
        )
        gene_pattern_q75 = moving_average(
            gene_pattern["q75"][:, gene_id].mean(axis=1), smoothing_window
        )
        times = gene_pattern["times"]

        if palette is not None and gp_name in palette:
            color = palette[gp_name]
        else:
            color = default_color_palette[i]
        if label_palette is not None and gp_name in label_palette:
            label = label_palette[gp_name]
        else:
            label = gp_name

        start_times.append(times[0])
        end_times.append(times[-1])

        plt.fill_between(
            times,
            gene_pattern_q25,
            gene_pattern_q75,
            color=color,
            alpha=0.3,
        )
        plt.plot(
            times,
            gene_pattern_mean,
            label=label,
            color=color,
            linewidth=3,
        )
    if crop_to_min_length:
        plt.xlim(max(start_times), min(end_times))
    else:
        plt.xlim(min(start_times), max(end_times))

    plt.xticks([])
    plt.xlabel("Decipher time", fontsize=14)
    plt.ylabel("Gene expression", fontsize=14)
    plt.ylim(0)
    plt.legend(frameon=False)
    plt.title(gene_name, fontsize=18)
