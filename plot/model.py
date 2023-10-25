import numpy as np
import scanpy as sc
from matplotlib import pyplot as plt


def plot_decipher_z(
    adata, basis="decipher_v_corrected", decipher_z_key_prefix="decipher_z", **kwargs
):

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


def plot_decipher_v(
    adata,
    color,
    title="",
    show_axis="arrow",
    figsize=(3.5, 3.5),
    palette=None,
    subsample_frac=1.0,
    x_label="Decipher 1",
    y_label="Decipher 2",
    ncols=2,
    ax_label_only_on_bottom_right=False,
    basis="decipher_v_corrected",
    **kwargs
):
    with plt.rc_context({"figure.figsize": figsize}):
        fig = sc.pl.embedding(
            sc.pp.subsample(adata, subsample_frac, copy=True),
            basis=basis,
            color=color,
            palette=palette,
            return_fig=True,
            frameon=(show_axis in [True, "arrow"]),
            ncols=ncols,
            **kwargs
        )
    ax = fig.axes[0]
    if type(color) == str or len(color) == 1:
        ax.set_title(title)

    # set labels and remove spines
    for i, ax in enumerate(fig.axes[::]):
        if ax._label == "<colorbar>":
            continue
        if show_axis == "arrow":
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False)
            ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False)

            if i % ncols == 0 or not ax_label_only_on_bottom_right:
                ax.set_ylabel(y_label)
            else:
                ax.set_ylabel(None)
            if i // ncols == (len(color) - 1) // ncols or not ax_label_only_on_bottom_right:
                ax.set_xlabel(x_label)
            else:
                ax.set_xlabel(None)
    return fig
