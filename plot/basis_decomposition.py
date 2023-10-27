from matplotlib import pyplot as plt


def basis(adata, colors=None, figsize=(5, 2.5), linewidth=3, ax=None):
    bases = adata.uns["decipher"]["basis_decomposition"]["basis"]
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    else:
        fig = ax.figure
    if type(linewidth) in [int, float]:
        linewidth = [linewidth] * bases.shape[1]
    for i in range(bases.shape[1]):
        ax.plot(
            bases[:, i],
            c=colors[i] if colors is not None else None,
            label="basis %d" % (i + 1),
            linewidth=linewidth[i],
        )
    plt.legend(loc="right", bbox_to_anchor=(1.35, 0.5), fancybox=False)
    return fig


def gene_patterns_decomposition(
    adata,
    gene_name,
    smoothing_window=5,
    cell_type_band_key=None,
    palette=None,
    label_palette=None,
    trajectory_names=None,
    figsize=(3, 2.3),
):
    pass
