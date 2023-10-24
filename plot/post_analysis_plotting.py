import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import torch


def plot_decipher_v(
    adata,
    color,
    title="",
    show_axis="arrow",
    figsize=(3.5, 3.5),
    palette=None,
    subsample_frac=1.0,
    basis="decipher_v_corrected",
    x_label="Decipher 1",
    y_label="Decipher 2",
    ax_label_only_on_edges=False,
    ncols=2,
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

    for i, ax in enumerate(fig.axes[::]):
        if ax._label == "<colorbar>":
            continue
        if show_axis == "arrow":
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False)
            ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False)

            if i % ncols == 0 or not ax_label_only_on_edges:
                ax.set_ylabel(y_label)
            else:
                ax.set_ylabel(None)
            if i // ncols == (len(color) - 1) // ncols or not ax_label_only_on_edges:
                ax.set_xlabel(x_label)
            else:
                ax.set_xlabel(None)
    return fig


def plot_trajectory(
    adata,
    trajectory_name,
    color="blue",
    ax=None,
):
    uns_key = "decipher_trajectories"
    checkpoints = adata.uns[uns_key][trajectory_name]["checkpoints"]
    if ax is None:
        ax = plt.gca()

    ax.plot(
        checkpoints[:, 0],
        checkpoints[:, 1],
        marker="o",
        c="black",
        markerfacecolor=color,
        markersize=7,
    )
    ax.plot(
        checkpoints[:1, 0],
        checkpoints[:1, 1],
        marker="*",
        markersize=20,
        c="black",
        markerfacecolor=color,
    )


def plot_gene_patterns(
    gene_name,
    adata,
    trajectories,
    gene_patterns,
    colors,
    labels=None,
    crop_to_equal_length=False,
    figsize=[3, 2.3],
):
    if type(gene_name) == list:
        gene_id = [adata.var_names.tolist().index(gn) for gn in gene_name]
    else:
        gene_id = adata.var_names.tolist().index(gene_name)

    def moving_average(x, w):
        if len(x.shape) == 2:
            return np.stack([moving_average(xx, w) for xx in x])
        return np.convolve(x, np.ones(w), "valid") / w

    singular = False
    if type(gene_patterns) != list:
        gene_patterns = [gene_patterns]
        singular = True
    if type(colors) != list:
        colors = [colors]
    if labels is None:
        labels = ""
    if type(labels) != list:
        labels = [labels]
    if type(trajectories) != list:
        trajectories = [trajectories]

    fig = plt.figure(figsize=figsize)
    min_time = 100000
    max_time = -100000
    if crop_to_equal_length:
        min_time, max_time = max_time, min_time

    for gene_pattern, color, label, trajectory in zip(
        gene_patterns,
        colors,
        labels,
        trajectories,
    ):
        if torch.is_tensor(gene_pattern):
            gene_pattern = gene_pattern.detach().numpy()
        if len(gene_pattern.shape) == 2:
            gene_pattern = gene_pattern[None, :, :]

        if type(gene_id) == list:
            gene_pattern = gene_pattern[:, :, gene_id].mean(axis=2)
        else:
            gene_pattern = gene_pattern[:, :, gene_id]
        gene_pattern = moving_average(gene_pattern, 5)
        gene_pattern_mean = gene_pattern.mean(axis=0)
        # gene_pattern_mean = moving_average(gene_pattern_mean, 5)
        gene_pattern_q25 = np.quantile(gene_pattern, 0.25, axis=0)
        gene_pattern_q75 = np.quantile(gene_pattern, 0.75, axis=0)
        # gene_pattern_q25 = moving_average(gene_pattern_q25, 5)
        # gene_pattern_q75 = moving_average(gene_pattern_q75, 5)

        x = trajectory.trajectory_time[2:-2]
        if crop_to_equal_length:
            min_time = max(x[0], min_time)
            max_time = min(x[-1], max_time)
        else:
            min_time = min(x[0], min_time)
            max_time = max(x[-1], max_time)

        plt.fill_between(
            x,
            gene_pattern_q25,
            gene_pattern_q75,
            color=color,
            alpha=0.3,
        )
        plt.plot(
            x,
            gene_pattern_mean,
            label=label,
            color=color,
            linewidth=3,
        )

    plt.xticks([])
    plt.xlabel("Decipher time", fontsize=14)
    plt.ylabel("Gene expression", fontsize=14)
    plt.ylim(0)
    plt.xlim(min_time, max_time)
    plt.legend(frameon=False)
    plt.title(gene_name, fontsize=18)


def add_cell_type_band(trajectory, palette):
    x = trajectory.trajectory_time
    plt.scatter(
        x,
        np.zeros(len(x)) - 0.032,
        c=[palette[c] for c in trajectory.cell_types],
        marker="s",
        s=20,
        transform=plt.gca().get_xaxis_transform(),
        clip_on=False,
        edgecolors=None,
    )
    plt.xlabel("Decipher time", fontsize=14, labelpad=10)
