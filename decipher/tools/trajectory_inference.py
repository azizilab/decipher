import itertools
import logging

import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.spatial
import torch.nn.functional
import elpigraph

from decipher.tools._decipher.data import decipher_load_model
from decipher.utils import create_decipher_uns_key

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
    """A class to configure the computation of a trajectory.

    The user can define a trajectory in different ways:
    - by providing the start and end clusters of the trajectory: `start_cluster_or_marker` and
      `end_cluster_or_marker`. These can be either cluster ids or marker genes. If they are marker
      genes, the cluster with the highest expression of the marker gene is selected. Then, the
      trajectory is computed as the shortest path between the two clusters in the minimum spanning
      tree of the clusters.
    - by providing a list of cluster ids: `cluster_ids_list`. Then, the trajectory is the path
      connecting the clusters in the order of the list.

    Parameters
    ----------
    name : str
        The name of the trajectory. It is used as a key to store the trajectory in the AnnData.
    start_cluster_or_marker : str, optional
        The start cluster id or marker gene. If it is a marker gene, the cluster with the highest
        expression of the marker gene is selected. If it is a cluster id, it is used as is.
    end_cluster_or_marker : str, optional
        The end cluster id or marker gene. If it is a marker gene, the cluster with the highest
        expression of the marker gene is selected. If it is a cluster id, it is used as is.
    subset_col : str, optional
        The column in `adata.obs` to subset on. If None, no subsetting is performed.
    subset_val : str, optional
        The value in subset_column to subset on. If None, no subsetting is performed.
    cluster_ids_list : list, optional
        A list of cluster ids. If provided, the trajectory is the path connecting the clusters in
        the order of the list. `start_cluster_or_marker` and `end_cluster_or_marker` are ignored.
    subset_percent_per_cluster : float, default 0.3
        When subsetting the cells, each cluster must have at least this proportion of cells from
        the subset to not be discarded. This is useful to remove clusters with too few cells from
        the subset.
    min_cell_per_cluster : int, default 10
        When subsetting the cells, each cluster must have at least this number of cells from the
        subset to not be discarded. See `subset_percent_per_cluster`.
    """

    def __init__(
        self,
        name: str,
        start_cluster_or_marker=None,
        end_cluster_or_marker=None,
        subset_col=None,
        subset_val=None,
        cluster_ids_list=None,
        subset_percent_per_cluster=0.3,
        min_cell_per_cluster=10,
    ):
        self.name = name
        self._start_cluster = start_cluster_or_marker
        self._end_cluster = end_cluster_or_marker
        self.subset_column = subset_col
        self.subset_value = subset_val
        self.subset_percent_per_cluster = subset_percent_per_cluster
        self.min_cell_per_cluster = min_cell_per_cluster
        self.cluster_ids_list = cluster_ids_list

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
            subset_min_percent_per_cluster=self.subset_percent_per_cluster,
            cluster_key=cluster_key,
            min_cell_per_cluster=self.min_cell_per_cluster,
        )
        return cluster_id

    def get_start_cluster_id(self, adata):
        return self._compute_cluster_id(adata, self._start_cluster)

    def get_end_cluster_id(self, adata):
        return self._compute_cluster_id(adata, self._end_cluster)


def cell_clusters(adata, leiden_resolution=1.0, n_neighbors=10, seed=0, rep_key="decipher_z"):
    """Compute the cell clusters using the scanpy Leiden algorithm on the Decipher z representation.

    Parameters
    ----------
    adata : sc.AnnData
        The annotated data matrix.
    leiden_resolution : float, default 1.0
        The resolution of the Leiden algorithm.
    n_neighbors : int, default 10
        The number of neighbors to use to compute the neighbors graph.
    seed : int, default 0
        The random seed to use in the scanpy functions.
    rep_key : str, default "decipher_z"
        The key in `adata.obsm` where the Decipher z representation is stored.

    Returns
    -------
    `adata.obs['decipher_clusters']`
        The cluster labels.

    See Also
    --------
    scanpy.pp.neighbors
    scanpy.tl.leiden
    """

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

    logging.info("Added `.obs['decipher_clusters']`: the cluster labels.")


def _subset_cells_and_clusters(
    adata,
    subset_column,
    subset_value,
    subset_min_percent_per_cluster=0.3,
    min_cell_per_cluster=10,
    cluster_key="decipher_clusters",
):
    data = adata.obs[[subset_column, cluster_key]].copy()
    data.columns = ["subset", "cluster"]
    data["in_subset"] = data["subset"] == subset_value
    # clusters that contains too few cells in the subset are discarded
    # 1) the proportion of cells from the subset in the cluster is too low
    valid_p = data.groupby("cluster")["in_subset"].mean() > subset_min_percent_per_cluster
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
    subset_min_percent_per_cluster=0.3,
    cluster_key="decipher_clusters",
    min_cell_per_cluster=10,
):
    """Find the cluster enriched for a marker gene. Possibly subset the cells before.

    Parameters
    ----------
    adata : sc.AnnData
        The annotated data matrix.
    marker : str
        The marker gene.
    subset_column : str, optional
        The column in `adata.obs` to subset on.
    subset_value : str, optional
        The value in subset_column to subset on.
    subset_min_percent_per_cluster : float, default 0.3
        When subsetting the cells, each cluster must have at least this proportion of cells from
        the subset to not be discarded. This is useful to remove clusters with too few cells from
        the subset.
    cluster_key : str, default "decipher_clusters"
        The key in `adata.obs` where the cluster information is stored.
    min_cell_per_cluster : int, default 10
        The minimum number of cells per cluster to consider it.
    """
    if subset_column is not None:
        adata = _subset_cells_and_clusters(
            adata,
            subset_column,
            subset_value,
            subset_min_percent_per_cluster=subset_min_percent_per_cluster,
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
    *trajectory_configs,
    eipi = False,
    point_density=50,
    numnodes=15,
    cluster_key="decipher_clusters",
):
    """Compute the trajectories given some trajectory configurations.

    Parameters
    ----------
    adata : sc.AnnData
        The annotated data matrix.
    trajectory_configs : TConfig
        The trajectory configurations.
    point_density : int, default 50
        The number of points per unit of length in the trajectory.
    cluster_key : str, default "decipher_clusters"
        The key in `adata.obs` where the cluster ids are stored.

    Returns
    -------
    `adata.uns["decipher"]["trajectories"]`: dict
        The trajectory inference results.
        - trajectory_name: dict
            - `cluster_locations`: np.ndarray (n_clusters, 2) - the locations of the clusters in
                the latent space.
            - `cluster_ids`: list of str - the cluster ids forming the trajectory
            - `points`: np.ndarray (n_points, 2) - the points of the trajectory in the latent space
                (where the number of points `n_point` in the trajectory depends on `point_density`)
            - `times`: np.ndarray (n_points,) - the times of the trajectory.
            - `density`: the density of points along the trajectory (= `point_density`)
    """
    rep_key = "decipher_v"
    uns_key = "trajectories"
    if eipi:
        cluster_key = 'cell_type'
    create_decipher_uns_key(adata)

    for t_config in trajectory_configs:
        t_name = t_config.name
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

        if eipi:
            cluster_locations = cluster_locations.loc[cluster_locations.index.isin(t_config.cluster_ids_list)].reindex(t_config.cluster_ids_list)
            data = cluster_locations[[0, 1]].values  # Assuming cluster_locations is a DataFrame with x, y columns
            
            # Run ElPiGraph to compute the principal graph
            curve = elpigraph.computeElasticPrincipalCurve(data, NumNodes=numnodes)
            nodes, edges = curve[0]['NodePositions'], curve[0]['Edges']

            t_cluster_ids = t_config.cluster_ids_list

            graph = nx.Graph()
            for i, (x, y) in cluster_locations[[0, 1]].iterrows():
                graph.add_node(i, x=x, y=y, xy=np.array([x, y]))

            for edge in edges[0]:
                graph.add_edge(edge[0], edge[1])

            # Find the trajectory order by traversing the graph
            # Assume the first node (node 0) is one end of the trajectory
            start_node = 0
            trajectory_order = list(nx.dfs_preorder_nodes(graph, source=start_node))

            # Sort nodes by the trajectory order
            nodes = nodes[trajectory_order]

            logging.info(f"Trajectory {t_name}: {t_cluster_ids}")
            t_cluster_locations = nodes.astype(np.float32)

        else:
            if t_config.cluster_ids_list is not None:
                t_cluster_ids = t_config.cluster_ids_list
            else:
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

            logging.info(f"Trajectory {t_name} : {t_cluster_ids})")
            t_cluster_locations = np.array(
                [cluster_locations.loc[cluster_id, [0, 1]] for cluster_id in t_cluster_ids]
            ).astype(np.float32)
        
        trajectory = Trajectory(
            t_cluster_locations,
            t_cluster_ids,
            point_density=point_density,
            rep_key=rep_key,
        )
        adata.uns["decipher"][uns_key][t_name] = trajectory.to_dict()
        logging.info(f"Added trajectory {t_name} to `adata.uns['decipher']['{uns_key}']`.")


def decipher_time(adata, n_neighbors=10):
    """Compute the decipher time for each cell, based on the inferred trajectories.

    The decipher time is computed by KNN regression of the cells' decipher v on the trajectories.

    Parameters
    ----------
    adata : sc.AnnData
        The annotated data matrix. The trajectories should have been computed and stored in
        `adata.uns["decipher"]["trajectories"]`.
    n_neighbors : int
        The number of neighbors to use for the KNN regression.

    Returns
    -------
    `adata.obs["decipher_time"]`
        The decipher time of each cell.
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

    logging.info("Added `.obs['decipher_time']`: the decipher time of each cell.")


def gene_patterns(adata, l_scale=10_000, n_samples=100):
    """Compute the gene patterns for each trajectory.

    The trajectories' points are sent through the decoders, thus defining distributions over the
    gene expression. The gene patterns are computed by sampling from these distribution.

    Parameters
    ----------
    adata : sc.AnnData
        The annotated data matrix. The trajectories should have been computed and stored in
        `adata.uns["decipher"]["trajectories"]`.
    l_scale : float
        The library size scaling factor.
    n_samples : int
        The number of samples to draw from the decoder to compute the gene pattern statistics.

    Returns
    -------
    `adata.uns["decipher"]["gene_patterns"]`: dict
        The gene patterns for each trajectory.
        - trajectory_name: dict
            - `mean`: the mean gene expression pattern
            - `q25`: the 25% quantile of the gene expression pattern
            - `q75`: the 75% quantile of the gene expression pattern
            - `times`: the times of the trajectory
    """
    uns_decipher_key = "gene_patterns"
    model = decipher_load_model(adata)
    if uns_decipher_key not in adata.uns["decipher"]:
        adata.uns["decipher"][uns_decipher_key] = dict()

    for t_name in adata.uns["decipher"]["trajectories"]:
        t_points = adata.uns["decipher"]["trajectories"][t_name]["points"]
        t_times = adata.uns["decipher"]["trajectories"][t_name]["times"]

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
