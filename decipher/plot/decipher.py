import numpy as np
import scanpy as sc
from matplotlib import pyplot as plt


def decipher_z(
    adata,
    basis="decipher_v",
    decipher_z_key="decipher_z",
    subset_of_zs=None,
    **kwargs,
):
    """Plot the Decipher v space colored by each dimension of the Decipher z space.

    Parameters
    ----------
    adata : sc.AnnData
        The annotated data matrix.
    basis : str, default "decipher_v"
        The basis to use for the plot.
    decipher_z_key : str, default "decipher_z"
        The key in `adata.obsm` where the decipher z space is stored.
    subset_of_zs : list of int, optional
        The dimensions of the decipher z space to plot. If None, plot all dimensions.
    **kwargs : dict, optional
        Additional arguments passed to `sc.pl.embedding`.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        The matplotlib figure.
    """
    dim_z = adata.obsm[decipher_z_key].shape[1]
    for i in range(dim_z):
        adata.obs["z%d" % (i + 1)] = adata.obsm[decipher_z_key][:, i]

    if subset_of_zs is None:
        subset_of_zs = list(range(1, dim_z + 1))

    return sc.pl.embedding(
        adata,
        basis=basis,
        color=[f"z{i}" for i in subset_of_zs],
        vmax=lambda xs: np.quantile(xs, 0.99),
        vmin=lambda xs: np.quantile(xs, 0.01),
        color_map="cool_r",
        frameon=False,
        show=False,
        sort_order=False,
        return_fig=True,
        **kwargs,
    )


def decipher(
    adata,
    color=None,
    palette=None,
    ncols=2,
    subsample_frac=1.0,
    title="",
    basis="decipher_v",
    x_label="Decipher 1",
    y_label="Decipher 2",
    axis_type="arrow",
    figsize=(3.5, 3.5),
    vmax=lambda xs: np.quantile(xs[~np.isnan(xs)], 0.99),
    **kwargs,
):
    """Plot the Decipher v space.

    Parameters
    ----------
    adata : sc.AnnData
        The annotated data matrix.
    color : str or list of str
        Keys for annotations of cells, given to `sc.pl.embedding`.
    palette : dict, optional
        A dictionary mapping color keys to colors.
    ncols : int, default 2
        Number of columns in the plot.
    subsample_frac : float, default 1.0
        Fraction of cells to plot. Useful for large datasets.
    title : str, default ""
        Title of the plot. Only used if `color` is a single key, otherwise the title for each
        subplot is set automatically to the name of the color key.
    basis : str, default "decipher_v"
        The basis to use for the plot.
    x_label : str, default "Decipher 1"
        The label for the x-axis.
    y_label : str, default "Decipher 2"
        The label for the y-axis.
    axis_type : str, default "arrow"
        The type of axis to use. Can be "arrow", "line", or "none".
        If "arrow", the axes are drawn as arrows, with no top or right spines.
        If "line", the axes are drawn as lines, with all spines.
        If "none", no axes are drawn.
    figsize : tuple, default (3.5, 3.5)
        The size of the figure.
    vmax : function, optional
        A function that takes a numpy array and returns a float. Used to set the maximum value of
        the colorbar. By default, the 99th percentile of the data is used.
    **kwargs : dict, optional
        Additional arguments passed to `sc.pl.embedding`.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        The matplotlib figure.

    See Also
    --------
    sc.pl.embedding

    """
    with plt.rc_context({"figure.figsize": figsize}):
        fig = sc.pl.embedding(
            sc.pp.subsample(adata, subsample_frac, copy=True),
            basis=basis,
            color=color,
            palette=palette,
            return_fig=True,
            frameon=(axis_type in ["line", "arrow"]),
            ncols=ncols,
            vmax=vmax if color is not None else None,
            **kwargs,
        )
    ax = fig.axes[0]
    if color is None or isinstance(color, str):
        color = [color]

    if len(color) == 1:
        ax.set_title(title)

    for i, ax in enumerate(fig.axes):
        if ax._label == "<colorbar>":
            continue
        if axis_type == "arrow":
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False)
            ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False)

        if axis_type != "none":
            if i % ncols == 0:
                ax.set_ylabel(y_label)
            else:
                ax.set_ylabel(None)
            if i // ncols == (len(color) - 1) // ncols:
                ax.set_xlabel(x_label)
            else:
                ax.set_xlabel(None)
    return fig
