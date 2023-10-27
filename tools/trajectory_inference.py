import itertools
import logging

import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.spatial
import torch.nn.functional

from model.data import load_decipher_model
from utils import create_decipher_uns_key

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s",
    level=logging.INFO,  # stream=sys.stdout
)


class Trajectory:
    def __init__(
        self,
        cluster_locations,
        cluster_ids,
        point_density=50,
        rep_key="decipher_v",
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
            rep_key=self.rep_key,
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


class TConfig:
    def __init__(
        self,
        start_cluster,
        end_cluster,
        subset_col=None,
        subset_val=None,
        subset_percent_per_cluster=0.3,
        min_cell_per_cluster=10,
    ):
        self._start_cluster = start_cluster
        self._end_cluster = end_cluster
        self.subset_column = subset_col
        self.subset_value = subset_val
        self.subset_percent_per_cluster = subset_percent_per_cluster
        self.min_cell_per_cluster = min_cell_per_cluster

    def _compute_cluster_id(self, adata, cluster_input, cluster_key="decipher_clusters"):
        # several things:
        # if cluster_input is present in `adata.obs[cluster_key]`,
        #   then it is a cluster id, so return it
        # otherwise,
        #   it is a marker gene, so find the cluster with the highest expression of this gene
        # for this, if subset_column is not None, subset the cells and remove clusters with too few
        # cells
        # then return the cluster id with the highest expression of the marker gene
        if cluster_input in adata.obs[cluster_key].values:
            # the cluster_input is already a cluster id
            return cluster_input

        # otherwise, it is a marker gene
        cluster_id = find_cluster_with_marker(
            adata,
            cluster_input,
            subset_column=self.subset_column,
            subset_value=self.subset_value,
            subset_percent_per_cluster=self.subset_percent_per_cluster,
            cluster_key=cluster_key,
            min_cell_per_cluster=self.min_cell_per_cluster,
        )
        return cluster_id

    def get_start_cluster_id(self, adata):
        return self._compute_cluster_id(adata, self._start_cluster)

    def get_end_cluster_id(self, adata):
        return self._compute_cluster_id(adata, self._end_cluster)


def cell_clusters(adata, leiden_resolution=1.0, n_neighbors=10, seed=0, rep_key="decipher_z"):

    logger.info("Clustering cells using scanpy Leiden algorithm.")
    neighbors_key = rep_key + "_neighbors"
    cluster_key = rep_key + "_clusters"
    logger.info(
        f"Step 1: Computing neighbors with `sc.pp.neighbors`, with the representation {rep_key}."
    )
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        random_state=seed,
        use_rep=rep_key,
        key_added=neighbors_key,
    )
    logger.info(
        f"Step 2: Computing clusters with `sc.tl.leiden`, and resolution {leiden_resolution}."
    )
    sc.tl.leiden(
        adata,
        random_state=seed,
        neighbors_key=neighbors_key,
        key_added=cluster_key,
        resolution=leiden_resolution,
    )
    adata.obs["decipher_clusters"] = pd.Categorical(adata.obs[cluster_key])


def _subset_cells_and_clusters(
    adata,
    subset_column,
    subset_value,
    subset_percent_per_cluster=0.3,
    min_cell_per_cluster=10,
    cluster_key="decipher_clusters",
):
    """ """
    data = adata.obs[[subset_column, cluster_key]].copy()
    data.columns = ["subset", "cluster"]
    data["in_subset"] = data["subset"] == subset_value
    # clusters that contains too few cells in the subset are discarded
    # 1) the proportion of cells from the subset in the cluster is too low
    valid_p = data.groupby("cluster").mean()["in_subset"] > subset_percent_per_cluster
    # 2) the number of cells from the subset in the cluster is too low
    valid_n = data.groupby("cluster")["in_subset"].sum() > min_cell_per_cluster
    valid_clusters = valid_p & valid_n
    valid_clusters = valid_clusters[valid_clusters].index

    cell_mask = data["cluster"].isin(valid_clusters)  #  & data["in_subset"]
    # TODO: #REPRODUCE# above, the commented part is for reproducibility of the main paper figures
    # will be added later once published (it does not change much)
    adata_subset = adata[cell_mask]
    return adata_subset


def find_cluster_with_marker(
    adata,
    marker,
    subset_column=None,
    subset_value=None,
    subset_percent_per_cluster=0.3,
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
        The column in `adata.obs` to subset on.
    subset_value : str (optional)
        The value in subset_column to subset on.
    cluster_key : str
        The key in `adata.obs` where the cluster information is stored.
    min_cell_per_cluster : int
        The minimum number of cells per cluster to consider it. If a cluster has fewer cells than
        this, it is discarded. It is especially useful when subsetting the cells.
    """
    if subset_column is not None:
        adata = _subset_cells_and_clusters(
            adata,
            subset_column,
            subset_value,
            subset_percent_per_cluster=subset_percent_per_cluster,
            min_cell_per_cluster=min_cell_per_cluster,
            cluster_key=cluster_key,
        )
    marker_data = pd.DataFrame(adata[:, marker].X.toarray())
    marker_data["cluster"] = adata.obs[cluster_key].values
    # get the proportion of cells in each cluster that are in the subset
    marker_data = marker_data.groupby("cluster").mean()
    marker_data = marker_data.sort_values(by=0, ascending=False)
    return marker_data.index[0]


def trajectories(
    adata,
    trajectories_config,
    resolution=50,
    cluster_key="decipher_clusters",
):
    """Compute the trajectories given some start and end clusters.

    Parameters
    ----------
    adata : AnnData
        The AnnData object.
    trajectories_config : dict
        A dictionary with TODO
    resolution : int
        The number of points per unit of length.
    cluster_key : str
        The key in `adata.obs` where the cluster ids are stored.

    Returns
    -------
    The trajectories are stored in `adata.uns["decipher"]["trajectories"]`, which is a nested
    dictionary with the structure:
    - trajectory_name
        - cluster_locations: the locations of the clusters in the latent space
        - cluster_ids: the ids of the clusters
        - points: the points of the trajectory in the latent space
        - times: the times of the trajectory
        - cumulative_length: the cumulative length of the trajectory
        - density: the density of points along the trajectory

    """
    rep_key = "decipher_v"
    uns_key = "trajectories"
    create_decipher_uns_key(adata)

    for t_name, t_config in trajectories_config.items():
        if t_config.subset_column is not None:
            adata_subset = _subset_cells_and_clusters(
                adata,
                t_config.subset_column,
                t_config.subset_value,
                t_config.subset_percent_per_cluster,
                t_config.min_cell_per_cluster,
                cluster_key=cluster_key,
            )
        else:
            adata_subset = adata

        cells_locations = pd.DataFrame(adata_subset.obsm[rep_key])
        cells_locations["cluster_id"] = adata_subset.obs[cluster_key].values

        valid_clusters = (
            cells_locations.groupby("cluster_id").count()[0] > t_config.min_cell_per_cluster
        )
        cluster_locations = cells_locations.groupby("cluster_id").mean()
        cluster_locations = cluster_locations[valid_clusters]

        graph = nx.Graph()
        for i, (x, y) in cluster_locations[[0, 1]].iterrows():
            graph.add_node(i, x=x, y=y, xy=np.array([x, y]))

        distance_func = scipy.spatial.distance.euclidean
        for n1, n2 in itertools.combinations(graph.nodes, r=2):
            n1_data, n2_data = graph.nodes[n1], graph.nodes[n2]
            distance = distance_func(n1_data["xy"], n2_data["xy"])
            graph.add_edge(n1, n2, weight=distance)
        tree = nx.minimum_spanning_tree(graph)

        start_cluster_id = t_config.get_start_cluster_id(adata)
        end_cluster_id = t_config.get_end_cluster_id(adata)

        t_cluster_ids = nx.shortest_path(tree, start_cluster_id, end_cluster_id)
        print(t_name, ":", t_cluster_ids)
        t_cluster_locations = np.array(
            [cluster_locations.loc[cluster_id, [0, 1]] for cluster_id in t_cluster_ids]
        ).astype(np.float32)
        trajectory = Trajectory(
            t_cluster_locations, t_cluster_ids, point_density=resolution, rep_key=rep_key
        )
        adata.uns["decipher"][uns_key][t_name] = trajectory.to_dict()


def decipher_time(adata, n_neighbors=10):
    """Compute the decipher time for each cell, based on the inferred trajectories.

    The decipher time is computed by KNN regression of the cells' decipher v on the trajectories.

    Parameters
    ----------
    adata : AnnData
        The AnnData object. The trajectories should have been computed and stored in
        `adata.uns["decipher"]["trajectories"]`.
    n_neighbors : int
        The number of neighbors to use for the KNN regression.
    """
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
            adata.obsm["decipher_v"][cells_on_trajectory_idx]
        )


def gene_patterns(adata, l_scale=10_000, n_samples=100):
    """Compute the gene patterns for each trajectory.

    The trajectories' points are sent through the decoders, thus defining distributions over the
    gene expression. The gene patterns are computed by sampling from these distribution.

    Parameters
    ----------
    adata : AnnData
        The AnnData object. The trajectories should have been computed and stored in
        `adata.uns["decipher"]["trajectories"]`.
    l_scale : float
        The library size scaling factor.
    n_samples : int
        The number of samples to draw from the decoder to compute the gene pattern statistics.

    Returns
    -------
    The gene patterns are stored in `adata.uns["decipher"]["gene_patterns"]`, which is a nested
    dictionary with the structure:
    - trajectory_name
        - mean: the mean gene expression pattern
        - q25: the 25% quantile of the gene expression pattern
        - q75: the 75% quantile of the gene expression pattern
        - times: the times of the trajectory
    """
    uns_decipher_key = "gene_patterns"
    model = load_decipher_model(adata)
    if uns_decipher_key not in adata.uns["decipher"]:
        adata.uns["decipher"][uns_decipher_key] = dict()

    for t_name in adata.uns["decipher"]["trajectories"]:
        t_points = adata.uns["decipher"]["trajectories"][t_name]["points"]
        t_times = adata.uns["decipher"]["trajectories"][t_name]["times"]

        # TODO: deprecated
        if "rotation" in adata.uns["decipher"]:
            # need to undo rotation of the decipher_v space
            t_points = t_points @ np.linalg.inv(adata.uns["decipher"]["rotation"])

        t_points = torch.FloatTensor(t_points)
        z_mean, z_scale = model.decoder_v_to_z(t_points)
        z_scale = torch.nn.functional.softplus(z_scale)

        z_samples = torch.distributions.Normal(z_mean, z_scale).sample(sample_shape=(n_samples,))

        gene_patterns = dict()
        gene_patterns["mean"] = (
            torch.nn.functional.softmax(model.decoder_z_to_x(z_mean), dim=-1).detach().numpy()
            * l_scale
        )

        gene_expression_samples = (
            torch.nn.functional.softmax(model.decoder_z_to_x(z_samples), dim=-1).detach().numpy()
            * l_scale
        )
        gene_patterns["q25"] = np.quantile(gene_expression_samples, 0.25, axis=0)
        gene_patterns["q75"] = np.quantile(gene_expression_samples, 0.75, axis=0)

        # Maybe this should be the mean that we use
        gene_patterns["mean2"] = gene_expression_samples.mean(axis=0)

        adata.uns["decipher"][uns_decipher_key][t_name] = {
            **gene_patterns,
            "times": t_times,
        }
