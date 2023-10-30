import logging

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

from decipher.plot.decipher import decipher as plot_decipher_v

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s",
    level=logging.INFO,  # stream=sys.stdout
)


def trajectories(
    adata,
    color=None,
    trajectory_names=None,
    palette=None,
):
    fig = plot_decipher_v(adata, color=color, palette=palette)
    ax = fig.axes[0]
    if trajectory_names is None:
        trajectory_names = adata.uns["decipher"]["trajectories"].keys()
    if type(trajectory_names) == str:
        trajectory_names = [trajectory_names]

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
    cell_type_key=None,
    palette=None,
    pattern_names=None,
    figsize=(3, 2.3),
    ax=None,
    include_uncertainty=True,
    max_length=None,
):
    if type(gene_name) == list:
        gene_id = [adata.var_names.tolist().index(gn) for gn in gene_name]
    else:
        gene_id = [adata.var_names.tolist().index(gene_name)]

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), "same") / np.convolve(np.ones_like(x), np.ones(w), "same")

    if pattern_names is None:
        pattern_names = list(adata.uns["decipher"]["gene_patterns"].keys())
    elif type(pattern_names) == str:
        pattern_names = [pattern_names]

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    else:
        fig = ax.figure
    start_times = []
    end_times = []

    default_color_palette = sns.color_palette(n_colors=len(pattern_names))
    for i, p_name in enumerate(pattern_names):
        gene_pattern = adata.uns["decipher"]["gene_patterns"][p_name]

        gene_pattern_mean = gene_pattern["mean"][:, gene_id].mean(axis=1)
        gene_pattern_mean = moving_average(gene_pattern_mean, smoothing_window)
        gene_pattern_q25 = gene_pattern["q25"][:, gene_id].mean(axis=1)
        gene_pattern_q25 = moving_average(gene_pattern_q25, smoothing_window)
        gene_pattern_q75 = gene_pattern["q75"][:, gene_id].mean(axis=1)
        gene_pattern_q75 = moving_average(gene_pattern_q75, smoothing_window)
        times = gene_pattern["times"]

        times = times[:max_length]
        gene_pattern_mean = gene_pattern_mean[:max_length]
        gene_pattern_q25 = gene_pattern_q25[:max_length]
        gene_pattern_q75 = gene_pattern_q75[:max_length]

        if palette is not None and p_name in palette:
            color = palette[p_name]
        else:
            color = default_color_palette[i]

        start_times.append(times[0])
        end_times.append(times[-1])

        if include_uncertainty:
            ax.fill_between(times, gene_pattern_q25, gene_pattern_q75, color=color, alpha=0.3)
        ax.plot(times, gene_pattern_mean, label=p_name, color=color, linewidth=3)
    if cell_type_key is not None:
        # TODO: need to decide which pattern_name to use for the cell type ... (all?)
        _add_cell_type_band(adata, pattern_names[-1], cell_type_key, ax, palette)

    if crop_to_min_length:
        ax.set_xlim(max(start_times), min(end_times))
    else:
        ax.set_xlim(min(start_times), max(end_times))

    ax.set_xticks([])
    ax.set_xlabel("Decipher time", fontsize=14)
    ax.set_ylabel("Gene expression", fontsize=14)
    ax.set_ylim(0)
    ax.legend(frameon=False)
    ax.set_title(gene_name, fontsize=18)

    return fig


def decipher_time(adata, **kwargs):
    return plot_decipher_v(adata, "decipher_time", **kwargs)


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
