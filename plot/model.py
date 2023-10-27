import numpy as np
import scanpy as sc
from matplotlib import pyplot as plt


def decipher_z(adata, basis="decipher_v_corrected", decipher_z_key_prefix="decipher_z", **kwargs):

    sc.pl.embedding(
        adata,
        basis="decipher_v_corrected",
        color=["z%d" % (i) for i in range(1, 11)],
        vmax=lambda xs: np.quantile(xs, 0.99),
        vmin=lambda xs: np.quantile(xs, 0.01),
        color_map="cool_r",
        frameon=False,
        show=False,
        sort_order=False,
        return_fig=True,
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
    **kwargs
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
            **kwargs
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
