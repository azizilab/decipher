import itertools

import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import torch

from sklearn.neighbors import KNeighborsClassifier

class Trajectory:
    def __init__(self, checkpoints, adata, cell_type_key=None, point_density=50):
        checkpoints = np.array(checkpoints)
        pairs = zip(checkpoints, checkpoints[1:])
        distances = []
        for s, e in pairs:
            v = e - s
            d = np.linalg.norm(v)
            distances.append(d)

        cumulative_length = np.cumsum(distances)

        self.cumulative_length = cumulative_length
        self.checkpoints = checkpoints
        self.n_points = int(point_density * self.cumulative_length[-1])

        self.trajectory_latent, self.trajectory_time = self._linspace(self.n_points)
        if cell_type_key is not None:
            self._compute_cell_types(adata, cell_type_key)

    def sample_uniformly(self, n_samples, seed=0):
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(self.trajectory_latent), size=n_samples)
        return self.trajectory_latent[indices], self.trajectory_time[indices]

    def _compute_cell_types(self, adata, cell_type_key):
        knc = KNeighborsClassifier(n_neighbors=50)
        knc.fit(adata.obsm["decipher_v_corrected"], adata.obs[cell_type_key])
        self.cell_types = knc.predict(self.trajectory_latent)

    def _linspace(self, num=100):
        total_length = self.cumulative_length[-1]
        times = np.linspace(0, total_length, num)
        res = []
        for t in times:
            res.append(self.at_time(t))
        trajectory_latent = np.array(res).astype(np.float32)
        trajectory_time = times

        return trajectory_latent, trajectory_time

    def at_percent(self, p):
        t = p * self.cumulative_length[-1]
        return self.at_time(t)

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

        return self.checkpoints[i] * (1 - t) + t * self.checkpoints[i + 1]


def rot(t):
    return np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])


def rotate_decipher_space(
    adata,
    label,
    decipher_component_to_align_label_with=1,
    flip_decipher_1=False,
    flip_latent_z=False,
):

    rot_scores = []

    adata_tmp = adata[~adata.obs[label].isna()]

    for t in np.linspace(0, 2 * np.pi):
        v_rotated = adata_tmp.obsm["decipher_v"] @ rot(t)
        decipher_1_score = np.corrcoef(v_rotated[:, 0], adata_tmp.obs[label])[1, 0]
        decipher_2_score = np.corrcoef(v_rotated[:, 1], adata_tmp.obs[label])[1, 0]
        if decipher_component_to_align_label_with == 1:
            rot_scores.append((decipher_1_score - np.abs(decipher_2_score), t))
        else:
            rot_scores.append((decipher_2_score - np.abs(decipher_1_score), t))

    best_t = max(rot_scores)[1]

    rotation = rot(best_t)
    if flip_decipher_1:
        rotation = rotation @ np.array([[-1,0], [0,1]])
    adata.obsm["decipher_v_corrected"] = adata.obsm["decipher_v"] @ rotation


    if flip_latent_z:
        # We want z to be correlated positively with the components
        z_sign_correction = np.sign(
            np.corrcoef(adata.obsm["decipher_z"], y=adata.obsm["decipher_v_corrected"], rowvar=False)[
                : adata.obsm["decipher_z"].shape[1], adata.obsm["decipher_z"].shape[1] :
            ].sum(axis=1)
        )
    else:
        z_sign_correction = 1.0

    adata.obsm["decipher_z_corrected"] = adata.obsm["decipher_z"] * z_sign_correction
    adata.uns["decipher_rotation"] = rotation


def cluster_representations(adata, leiden_resolution=1.0, seed=0):
    sc.pp.neighbors(
        adata,
        n_neighbors=10,
        random_state=seed,
        use_rep="decipher_z_corrected",
        key_added="cluster_z",
    )
    sc.tl.leiden(
        adata,
        random_state=seed,
        neighbors_key="cluster_z",
        key_added="leiden_decipher_z",
        resolution=leiden_resolution,
    )
    adata.obs["PhenoGraph_clusters"] = pd.Categorical(adata.obs["leiden_decipher_z"])


def compute_trajectories(
    adata,
    start_marker,
    end_marker,
    subset_column=None,
    subset_value=None,
    trajectory_point_density=50,
    cell_types_key=None,
):
    cells_location = pd.DataFrame(adata.obsm["decipher_v_corrected"])
    cells_location["cluster"] = adata.obs["PhenoGraph_clusters"].values
    markers = [start_marker, end_marker]

    for m in markers:
        cells_location[m] = adata[:, m].X.toarray().reshape(-1)

    aggregation = {
        0: "mean",
        1: "mean",
        **dict.fromkeys(markers, "mean"),
    }
    if subset_column is not None:
        aggregation[subset_column] = pd.Series.mode
        cells_location[subset_column] = adata.obs[subset_column].values

    cluster_data = cells_location.groupby("cluster").agg(aggregation)
    if subset_column is not None:
        cluster_data = cluster_data[cluster_data[subset_column] == subset_value]

    G = nx.Graph()
    for i, (x, y) in cluster_data[[0, 1]].iterrows():
        G.add_node(i, x=x, y=y, xy=np.array([x, y]))

    distance_func = scipy.spatial.distance.euclidean
    for n1, n2 in itertools.combinations(G.nodes, r=2):
        n1_data, n2_data = G.nodes[n1], G.nodes[n2]
        distance = distance_func(n1_data["xy"], n2_data["xy"])
        G.add_edge(n1, n2, weight=distance)

    tree = nx.minimum_spanning_tree(G)

    start = cluster_data[start_marker].sort_values().index[-1]
    end = cluster_data[end_marker].sort_values().index[-1]
    trajectory_nodes = nx.shortest_path(tree, start, end)

    trajectory_nodes_location = np.array(
        [cluster_data.loc[cluster_id, [0, 1]] for cluster_id in trajectory_nodes]
    )

    trajectory_length = 0
    for i in range(1, len(trajectory_nodes_location)):
        trajectory_length += distance_func(trajectory_nodes_location[i - 1], trajectory_nodes_location[i])

    trajectory = Trajectory(trajectory_nodes_location, adata,  cell_types_key, trajectory_point_density,)

    return trajectory

def compute_decipher_time(adata, trajectories, n_neighbors=10):
    from sklearn.neighbors import KNeighborsRegressor
    adata.obs["origin_cat"] = adata.obs["origin"].astype("category")
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    Xs = []
    Ts = []
    for origin in trajectories:
        X_tmp = trajectories[origin].trajectory_latent @ np.array([[1,0,0], [0,1,0]])
        X_tmp[:,2] = adata.obs["origin_cat"].cat.categories.tolist().index(origin)*1000
        Xs.append(X_tmp)
        Ts.append(trajectories[origin].trajectory_time)
    X = np.concatenate(Xs)
    t = np.concatenate(Ts)
    knn.fit(X, t)
    Xt = adata.obsm["decipher_v_corrected"] @ np.array([[1,0,0], [0,1,0]])
    Xt[:, 2] = adata.obs["origin_cat"].cat.codes.values*1000
    adata.obs["decipher_time"] = knn.predict(Xt)


def sample_from_decipher_trajectory(adata, model, trajectory_latent, l_scale=10_000, n_samples=5, smooth=0, return_mean=False):
    if "decipher_rotation" in adata.uns:
        # need to undo rotation of the decipher_v_corrected space
        trajectory_latent = trajectory_latent @ np.linalg.inv(adata.uns["decipher_rotation"])

    trajectory_latent = torch.FloatTensor(trajectory_latent)
    z_mean, z_scale = model.decoder_p_to_z(trajectory_latent)
    z_scale = torch.nn.functional.softplus(z_scale)

    if return_mean:
        samples_z = z_mean
    else:
        if smooth == 0:
            samples_z = torch.distributions.Normal(z_mean, z_scale).sample(
                sample_shape=(n_samples,)
            )
        else:
            from scipy.linalg import cholesky as compute_cholesky
            dim = len(trajectory_latent)
            covx, covy = np.meshgrid(np.arange(dim), np.arange(dim))
            L = smooth
            cov = np.exp(-(covx - covy) ** 2 / (2 * L * L))
            cov += np.eye(dim) * 1e-10
            cov_cholesky = compute_cholesky(cov, lower=True)

            choleskies_correlation = torch.FloatTensor(cov_cholesky)
            choleskies_covariance = torch.einsum(
                "id,ij -> ijd", z_scale, choleskies_correlation
            )
            samples_normal = torch.distributions.Normal(0.0, 1.0).sample(
                sample_shape=(n_samples,) + z_mean.shape
            )
            samples_z = z_mean + torch.einsum(
                "sjd,ijd->sid", samples_normal, choleskies_covariance
            )

    samples_genes = torch.nn.functional.softmax(model.decoder_z_to_x(samples_z), dim=-1) * l_scale
    return samples_genes.detach()


#     fig = sc.pl.embedding(
#         adata,
#         basis="decipher_v_corrected",
#         color=["cell_type_merged"],
#         frameon=False,
#         show=False,
#         return_fig=True,
#     )

#     nx.draw_networkx(
#         tree,
#         nx.get_node_attributes(tree, "xy"),
#         node_color="red",
#     )

#     fig.axes[0].plot(
#         (trajectory_nodes_location)[:, 0],
#         (trajectory_nodes_location)[:, 1],
#         marker="o",
#         c="black",
#         markerfacecolor="orange",
#     )
#     fig.axes[0].plot(
#         (trajectory_nodes_location)[:1, 0],
#         (trajectory_nodes_location)[:1, 1],
#         marker="*",
#         markersize=20,
#         c="black",
#         markerfacecolor="orange",
#     )
