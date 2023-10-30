from matplotlib import pyplot as plt
import seaborn as sns


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
    ax.legend(loc="right", bbox_to_anchor=(1.35, 0.5), fancybox=False)
    return fig


from .trajectory import gene_patterns as plot_gene_patterns


def gene_patterns_decomposition(
    adata,
    gene_name,
    pattern_name,
    palette=None,
    basis_colors=None,
    figsize=(7, 2),
):
    gene_id = adata.var_names.tolist().index(gene_name)
    pattern_id = adata.uns["decipher"]["basis_decomposition"]["pattern_names"].index(pattern_name)
    max_length = adata.uns["decipher"]["basis_decomposition"]["length"]
    fig, axes = plt.subplots(1, 3, gridspec_kw={"width_ratios": [3, 3, 1]}, figsize=figsize)

    plot_gene_patterns(
        adata,
        gene_name,
        include_uncertainty=False,
        ax=axes[0],
        palette=palette,
        pattern_names=pattern_name,
        max_length=max_length,
    )

    beta = adata.uns["decipher"]["basis_decomposition"]["betas"][pattern_id, gene_id]
    n_basis = beta.shape[0]
    if basis_colors is None:
        basis_colors = sns.color_palette("Accent", n_colors=n_basis)
    basis(adata, colors=basis_colors, linewidth=beta * 10, ax=axes[1])
    axes[1].set_yticks([])
    axes[1].get_legend().remove()

    ax = axes[2]
    ax.barh(y=range(n_basis)[::-1], width=beta, color=basis_colors)

    ax.set_yticks(range(n_basis)[::-1])
    ax.set_xlim(0, 1)
    ax.set_yticklabels([f"$\\beta_%d$" % i for i in range(n_basis)], fontsize=12)

    for ax in axes:
        ax.set_xticks([])

    return fig
