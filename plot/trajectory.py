import logging

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

from plot.model import decipher as plot_decipher_v

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s",
    level=logging.INFO,  # stream=sys.stdout
)

# def cell_clusters(
#     adata,
#     color="decipher_cluster",
#     show_tree=True,
#     palette=None,
#     ax=None,
# ):
#     plot_decipher_v(adata, color, palette=palette, ax=ax)


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
    cell_type_band_key=None,
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
    ax = fig.gca()
    start_times = []
    end_times = []

    default_color_palette = sns.color_palette(n_colors=len(gene_patterns_names))
    for i, gp_name in enumerate(gene_patterns_names):
        gene_pattern = adata.uns["decipher"]["gene_patterns"][gp_name]

        gene_pattern_mean = gene_pattern["mean"][:, gene_id].mean(axis=1)
        gene_pattern_mean = moving_average(gene_pattern_mean, smoothing_window)
        gene_pattern_q25 = gene_pattern["q25"][:, gene_id].mean(axis=1)
        gene_pattern_q25 = moving_average(gene_pattern_q25, smoothing_window)
        gene_pattern_q75 = gene_pattern["q75"][:, gene_id].mean(axis=1)
        gene_pattern_q75 = moving_average(gene_pattern_q75, smoothing_window)
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

        ax.fill_between(times, gene_pattern_q25, gene_pattern_q75, color=color, alpha=0.3)
        ax.plot(times, gene_pattern_mean, label=label, color=color, linewidth=3)
    if cell_type_band_key is not None:
        _add_cell_type_band(adata, gene_patterns_names[0], cell_type_band_key, ax, palette)

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


def decipher_time(adata, **kwargs):
    plot_decipher_v(adata, "decipher_time", **kwargs)


def _add_cell_type_band(adata, trajectory_name, cell_type_key, ax, palette, n_neighbors=50):
    trajectory = adata.uns["decipher"]["trajectories"][trajectory_name]
    if (
        "cell_types" not in trajectory
        or trajectory["cell_types"]["key"] != cell_type_key
        or (trajectory["cell_types"]["n_neighbors"] != n_neighbors)
    ):
        knc = KNeighborsClassifier(n_neighbors=n_neighbors)
        knc.fit(adata.obsm[trajectory["rep_key"]], adata.obs[cell_type_key])
        cell_types = knc.predict(trajectory["points"])
        trajectory["cell_types"] = {
            "key": cell_type_key,
            "n_neighbors": n_neighbors,
            "values": cell_types,
        }

    times = trajectory["times"]
    cell_types = trajectory["cell_types"]["values"]

    if palette is None:
        logging.info("No palette provided for the cell types, using default.")
        ct = np.unique(cell_types)
        palette = dict(zip(ct, sns.color_palette(n_colors=len(ct))))

    plt.scatter(
        times,
        np.zeros(len(times)) - 0.03,
        c=[palette[c] for c in cell_types],
        marker="s",
        s=20,
        transform=plt.gca().get_xaxis_transform(),
        clip_on=False,
        edgecolors=None,
    )
    ax.xaxis.labelpad = 10
