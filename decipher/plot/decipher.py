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
    if color is None or type(color) == str:
        color = [color]

    if len(color) == 1:
        ax.set_title(title)

    # set labels and remove spines
    for i, ax in enumerate(fig.axes):
        if ax._label == "<colorbar>":
            continue
        if axis_type == "arrow":
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False)
            ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False)

            if i % ncols == 0:
                ax.set_ylabel(y_label)
            else:
                ax.set_ylabel(None)
            if i // ncols == (len(color) - 1) // ncols:
                ax.set_xlabel(x_label)
            else:
                ax.set_xlabel(None)
    return fig
