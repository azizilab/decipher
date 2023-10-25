from matplotlib import pyplot as plt
import seaborn as sns


def plot_trajectory(
    adata,
    trajectory_names=None,
    palette=None,
    ax=None,
):
    if trajectory_names is None:
        trajectory_names = adata.uns["decipher"]["trajectories"].keys()
    if type(trajectory_names) == str:
        trajectory_names = [trajectory_names]
    if ax is None:
        ax = plt.gca()

    for i, t_name in enumerate(trajectory_names):
        cluster_locations = adata.uns["decipher"]["trajectories"][t_name]["cluster_locations"]
        if palette is not None:
            color = palette[t_name]
        else:
            color = sns.color_palette()[i]
        ax.plot(
            cluster_locations[:, 0],
            cluster_locations[:, 1],
            marker="o",
            c="black",
            markerfacecolor=color,
            markersize=7,
        )
        ax.plot(
            cluster_locations[:1, 0],
            cluster_locations[:1, 1],
            marker="*",
            markersize=20,
            c="black",
            markerfacecolor=color,
        )
