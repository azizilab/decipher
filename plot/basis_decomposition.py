from matplotlib import pyplot as plt


def basis(adata, colors=None):
    bases = adata.uns["decipher"]["basis_decomposition"]["basis"]
    fig = plt.figure(figsize=(5, 2.5))
    for i in range(bases.shape[1]):
        plt.plot(
            bases[:, i],
            c=colors[i] if colors is not None else None,
            label="basis %d" % (i + 1),
            linewidth=3,
        )
    plt.legend(loc="right", bbox_to_anchor=(1.35, 0.5), fancybox=False)
