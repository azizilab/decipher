import logging

import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from decipher.plot.decipher import decipher as plot_decipher_v

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s",
    level=logging.INFO,
)


def cell_clusters(adata, legend_loc="on data", legend_fontsize=10):
    return plot_decipher_v(
        adata,
        color="decipher_clusters",
        legend_loc=legend_loc,
        legend_fontsize=legend_fontsize,
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
    if isinstance(trajectory_names, str):
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
    gene_names,
    crop_to_min_length=False,
    smoothing_window=5,
    cell_type_key=None,
    palette=None,
    pattern_names=None,
    figsize=(30, 20),
    include_uncertainty=True,
    max_length=None,
    cell_type_band_pattern_names=None,
    nrows=5,
    ncols=4,
):
    """Plot the gene patterns over the Decipher time in a grid of subplots.

    Parameters
    ----------
    adata : sc.AnnData
        The annotated data matrix.
    gene_names : list of str
        The names of the genes to plot.
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
    figsize : tuple of float, default (15, 10)
        The size of the figure.
    include_uncertainty : bool, default True
        Whether to include the uncertainty of the gene patterns in the plot, as a shaded area.
    max_length : int, optional
        The maximum length of the gene patterns to plot. If None, plot the full length, up to the
        minimum length of the gene patterns if `crop_to_min_length` is True.
    cell_type_band_pattern_names : str or list of str, optional
        The names of the gene patterns to use for the cell type bands. If None, use the same gene
        patterns as `pattern_names`. It is useful to use a subset of the gene patterns to avoid
        multiple bands.
    nrows : int, default 2
        Number of rows in the subplot grid.
    ncols : int, default 2
        Number of columns in the subplot grid.
    """

    # Ensure gene_names is a list
    if isinstance(gene_names, str):
        gene_names = [gene_names]

    # Create figure and axes for subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()  # Flatten in case nrows * ncols > 1

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), "same") / np.convolve(np.ones_like(x), np.ones(w), "same")

    if pattern_names is None:
        pattern_names = list(adata.uns["decipher"]["gene_patterns"].keys())
    elif isinstance(pattern_names, str):
        pattern_names = [pattern_names]

    default_color_palette = sns.color_palette(n_colors=len(pattern_names))

    for i, gene_name in enumerate(gene_names):
        if i >= len(axes):
            break  # If there are more genes than subplots, exit

        gene_id = [adata.var_names.tolist().index(gene_name)]
        ax = axes[i]  # Select the appropriate subplot

        start_times = []
        end_times = []

        for j, p_name in enumerate(pattern_names):
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
                color = default_color_palette[j]

            start_times.append(times[0])
            end_times.append(times[-1])

            if include_uncertainty:
                ax.fill_between(times, gene_pattern_q25, gene_pattern_q75, color=color, alpha=0.3)
            ax.plot(times, gene_pattern_mean, label=p_name, color=color, linewidth=2)

        if cell_type_key is not None:
            if cell_type_band_pattern_names is None:
                cell_type_band_pattern_names = pattern_names
            for j, p_name in enumerate(cell_type_band_pattern_names):
                _add_cell_type_band(adata, p_name, cell_type_key, ax, palette, offset=j + 1)

        if crop_to_min_length:
            ax.set_xlim(max(start_times), min(end_times))
        else:
            ax.set_xlim(min(start_times), max(end_times))

        ax.set_xticks([])
        ax.set_xlabel("Decipher time", fontsize=10)
        ax.set_ylabel("Gene expression", fontsize=10)
        ax.set_ylim(0)
        ax.set_title(gene_name, fontsize=12)
        ax.legend(frameon=False, fontsize=8)

    # Remove unused axes if gene_names < nrows * ncols
    for i in range(len(gene_names), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
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
    adata, trajectory_name, cell_type_key, ax, palette=None, n_neighbors=50, offset=1
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

    # Use the ax provided instead of the default plt
    ax.scatter(
        times,
        np.zeros(len(times)) - 0.05 * offset + 0.025,
        c=[palette[c] for c in cell_types],
        marker="s",
        s=20,
        transform=ax.get_xaxis_transform(),
        clip_on=False,
        edgecolors=None,
    )
    ax.xaxis.labelpad = 5 + 5 * offset
