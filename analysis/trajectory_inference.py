import itertools

import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.spatial
from sklearn.neighbors import KNeighborsClassifier


def create_decipher_uns_key(adata):
    if "decipher" not in adata.uns:
        adata.uns["decipher"] = dict()
    if "trajectories" not in adata.uns["decipher"]:
        adata.uns["decipher"]["trajectories"] = dict()


class Trajectory:
    def __init__(
        self,
        cluster_locations,
        cluster_ids,
        point_density=50,
        rep_key="decipher_v_corrected",
    ):
        self._point_density = point_density
        self.cluster_ids = cluster_ids
        cluster_locations = np.array(cluster_locations)
        distances = []
        for s, e in zip(cluster_locations, cluster_locations[1:]):
            v = e - s
            d = np.linalg.norm(v)
            distances.append(d)

        cumulative_length = np.cumsum(distances)

        self.cumulative_length = cumulative_length
        self.cluster_locations = cluster_locations
        self.n_points = int(self._point_density * self.cumulative_length[-1])
        self.rep_key = rep_key

        self.trajectory_latent, self.trajectory_time = self._linspace(self.n_points)

    # def sample_uniformly(self, n_samples, seed=0):
    #     rng = np.random.default_rng(seed)
    #     indices = rng.integers(0, len(self.trajectory_latent), size=n_samples)
    #     return self.trajectory_latent[indices], self.trajectory_time[indices]

    # def _compute_cell_types(self, adata, cell_type_key, n_neighbors=50):
    #     knc = KNeighborsClassifier(n_neighbors=n_neighbors)
    #     knc.fit(adata.obsm[self.rep_key], adata.obs[cell_type_key])
    #     self.cell_types = knc.predict(self.trajectory_latent)

    def _linspace(self, num=100):
        total_length = self.cumulative_length[-1]
        times = np.linspace(0, total_length, num)
        res = []
        for t in times:
            res.append(self.at_time(t))
        trajectory_latent = np.array(res).astype(np.float32)
        trajectory_time = times

        return trajectory_latent, trajectory_time

    def at_time(self, t):
        i = 0
        while t > self.cumulative_length[i]:
            i += 1
        if i > 0:
            t = (t - self.cumulative_length[i - 1]) / (
                self.cumulative_length[i] - self.cumulative_length[i - 1]
            )
        else:
            t = t / self.cumulative_length[i]

        return self.cluster_locations[i] * (1 - t) + t * self.cluster_locations[i + 1]

    def to_dict(self):
        return dict(
            cluster_locations=self.cluster_locations,
            cluster_ids=self.cluster_ids,
            points=self.trajectory_latent,
            times=self.trajectory_time,
            cumulative_length=self.cumulative_length,
            density=self._point_density,
        )

    @staticmethod
    def from_dict(d):
        trajectory = Trajectory(
            d["cluster_locations"],
            d["cluster_ids"],
            point_density=d["density"],
            rep_key=d["rep_key"],
        )
        return trajectory


def cluster_cells(adata, leiden_resolution=1.0, seed=0, rep_key="decipher_z_corrected"):
    neighbors_key = rep_key + "_neighbors"
    cluster_key = rep_key + "_clusters"
    sc.pp.neighbors(
        adata,
        n_neighbors=10,
        random_state=seed,
        use_rep=rep_key,
        key_added=neighbors_key,
    )
    sc.tl.leiden(
        adata,
        random_state=seed,
        neighbors_key=neighbors_key,
        key_added=cluster_key,
        resolution=leiden_resolution,
    )
    adata.obs["decipher_clusters"] = pd.Categorical(adata.obs[cluster_key])


def find_cluster_with_marker(
    adata,
    marker,
    subset_column=None,
    subset_value=None,
    cluster_key="decipher_clusters",
    min_cell_per_cluster=10,
):
    """Find the cluster enriched for a marker gene. Possibly subset the cells before.

    Parameters
    ----------
    adata : AnnData
        The AnnData object.
    marker : str
        The marker gene.
    subset_column : str (optional)
        The column in adata.obs to subset on.
    subset_value : str (optional)
        The value in subset_column to subset on.
    cluster_key : str
        The key in adata.obs where the cluster information is stored.
    """
    marker_data = pd.DataFrame(adata[:, marker].X.toarray())
    marker_data["cluster"] = adata.obs[cluster_key].values
    if subset_column is not None:
        marker_data["subset"] = adata.obs[subset_column].values
        marker_data = marker_data[marker_data["subset"] == subset_value]
    valid_cluster = marker_data.groupby("cluster").count()[0] > min_cell_per_cluster
    marker_data = marker_data.groupby("cluster").mean()
    marker_data = marker_data[valid_cluster]
    marker_data = marker_data.sort_values(by=0, ascending=False)
    return marker_data.index[0]


def compute_trajectories(
    adata,
    start_end_clusters_dict,
    resolution=50,
    cluster_key="decipher_clusters",
):
    """Compute the trajectories given some start and end clusters.

    Parameters
    ----------
    adata : AnnData
        The AnnData object.
    start_end_clusters_dict : dict
        A dictionary with the name of the trajectory as key and a tuple of the start and end cluster
        as value.
    resolution : int
        The number of points per unit of length.
    cluster_key : str
        The key in adata.obs where the cluster ids are stored.

    Returns
    -------
    The trajectories are stored in adata.uns["decipher"]["trajectories"], which is a nested
    dictionary with the structure:
    - trajectory_name
        - cluster_locations: the locations of the clusters in the latent space
        - cluster_ids: the ids of the clusters
        - points: the points of the trajectory in the latent space
        - times: the times of the trajectory
        - cumulative_length: the cumulative length of the trajectory
        - density: the density of points along the trajectory

    """
    rep_key = "decipher_v_corrected"
    uns_key = "trajectories"

    cells_locations = pd.DataFrame(adata.obsm[rep_key])
    cells_locations["cluster_id"] = adata.obs[cluster_key].values
    cluster_locations = cells_locations.groupby("cluster_id").mean()

    graph = nx.Graph()
    for i, (x, y) in cluster_locations[[0, 1]].iterrows():
        graph.add_node(i, x=x, y=y, xy=np.array([x, y]))

    distance_func = scipy.spatial.distance.euclidean
    for n1, n2 in itertools.combinations(graph.nodes, r=2):
        n1_data, n2_data = graph.nodes[n1], graph.nodes[n2]
        distance = distance_func(n1_data["xy"], n2_data["xy"])
        graph.add_edge(n1, n2, weight=distance)

    tree = nx.minimum_spanning_tree(graph)

    create_decipher_uns_key(adata)
    for t_name, (start_cluster, end_cluster) in start_end_clusters_dict.items():
        t_cluster_ids = nx.shortest_path(tree, start_cluster, end_cluster)
        print(t_name, ":", t_cluster_ids)
        t_cluster_locations = np.array(
            [cluster_locations.loc[cluster_id, [0, 1]] for cluster_id in t_cluster_ids]
        ).astype(np.float32)
        trajectory = Trajectory(
            t_cluster_locations, t_cluster_ids, point_density=resolution, rep_key=rep_key
        )
        adata.uns["decipher"][uns_key][t_name] = trajectory.to_dict()


def compute_decipher_time(adata, n_neighbors=10):
    from sklearn.neighbors import KNeighborsRegressor

    cluster_key = "decipher_clusters"
    adata.obs["decipher_time"] = np.nan
    for name, trajectory in adata.uns["decipher"]["trajectories"].items():
        knn = KNeighborsRegressor(n_neighbors=n_neighbors)
        knn.fit(trajectory["points"], trajectory["times"])
        is_on_trajectory = adata.obs[cluster_key].isin(trajectory["cluster_ids"])
        cells_on_trajectory_index = adata.obs[is_on_trajectory].index
        cells_on_trajectory_idx = np.where(is_on_trajectory)[0]

        adata.obs.loc[cells_on_trajectory_index, "decipher_time"] = knn.predict(
            adata.obsm["decipher_v_corrected"][cells_on_trajectory_idx]
        )


def compute_gene_patterns(adata):
    pass
