import logging

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

from decipher.plot.decipher import decipher as plot_decipher_v

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s",
    level=logging.INFO,  # stream=sys.stdout
)


def cell_clusters(adata, legend_loc="on data", legend_fontsize=10):
    return plot_decipher_v(
        adata, color="decipher_clusters", legend_loc=legend_loc, legend_fontsize=legend_fontsize
    )


def trajectories(
    adata,
    color=None,
    trajectory_names=None,
    palette=None,
):
    """Plot the trajectories over the Decipher v space.

    Parameters
    ----------
    adata : sc.AnnData
        The annotated data matrix.
    color : str, optional
        Key (or list of keys) for color annotations of cells, passed to `dc.pl.decipher` which
        in turn passes it to `sc.pl.embedding`. The keys should be in `adata.obs` or should be
        gene names.
    trajectory_names : str or list of str, optional
        The names of the trajectories to plot. If None, plot all trajectories.
    palette : dict, optional
        A dictionary mapping trajectory names to colors and/or mapping color keys to colors.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        The matplotlib figure.

    See Also
    --------
    dc.pl.decipher
    sc.pl.embedding
    """
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
            c=color,
            markerfacecolor=color,
            markersize=7,
            path_effects=[pe.Stroke(linewidth=3, foreground="black"), pe.Normal()],
        )
        ax.plot(
            cluster_locations[:1, 0],
            cluster_locations[:1, 1],
            marker="*",
            markersize=20,
            c="black",
            markerfacecolor=color,
        )

    return fig


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
    cell_type_band_pattern_names=None,
):
    """Plot the gene patterns over the Decipher time.

    Parameters
    ----------
    adata : sc.AnnData
        The annotated data matrix.
    gene_name : str or list of str
        The name(s) of the gene(s) to plot.
    crop_to_min_length : bool, default False
        Crop the plot to the minimum length of the gene patterns.
    smoothing_window : int, default 5
        The size of the window for the moving average smoothing.
    cell_type_key : str, optional
        The key of the cell type annotations in `adata.obs`. If provided, the cell types will be
        plotted as colored bands on the x-axis. See `cell_type_band_pattern_names` for more details.
    palette : dict, optional
        A dictionary mapping pattern names and cell type names to colors.
    pattern_names : str or list of str, optional
        The names of the gene patterns to plot. If None, plot all gene patterns.
    figsize : tuple of float, default (3, 2.3)
        The size of the figure.
    ax : matplotlib.pyplot.Axes, optional
        The axes on which to plot. If None, create a new figure and axes.
    include_uncertainty : bool, default True
        Whether to include the uncertainty of the gene patterns in the plot, as a shaded area.
    max_length : int, optional
        The maximum length of the gene patterns to plot. If None, plot the full length, up to the
        minimum length of the gene patterns if `crop_to_min_length` is True.
    cell_type_band_pattern_names : str or list of str, optional
        The names of the gene patterns to use for the cell type bands. If None, use the same gene
        patterns as `pattern_names`. It is useful to use a subset of the gene patterns to avoid
        multiple bands.
    """
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
        if cell_type_band_pattern_names is None:
            cell_type_band_pattern_names = pattern_names
        for i, p_name in enumerate(cell_type_band_pattern_names):
            _add_cell_type_band(adata, p_name, cell_type_key, ax, palette, offset=i + 1)

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
    """Plot the Decipher time over the Decipher v space.

    Parameters
    ----------
    adata : sc.AnnData
        The annotated data matrix.
    **kwargs : dict
        Keyword arguments passed to `dc.pl.decipher` and ultimately to `sc.pl.embedding`.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        The matplotlib figure.
    """
    return plot_decipher_v(adata, "decipher_time", **kwargs)


def _add_cell_type_band(
    adata, trajectory_name, cell_type_key, ax, palette, n_neighbors=50, offset=1
):
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
        np.zeros(len(times)) - 0.05 * offset + 0.025,
        c=[palette[c] for c in cell_types],
        marker="s",
        s=20,
        transform=plt.gca().get_xaxis_transform(),
        clip_on=False,
        edgecolors=None,
    )
    ax.xaxis.labelpad = 5 + 5 * offset
