import itertools

import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import torch

from sklearn.neighbors import KNeighborsClassifier


class Trajectory:
    def __init__(self, checkpoints, adata, cell_type_key=None, point_density=50, rep_key="decipher_v_corrected"):
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
        self.rep_key = rep_key

        self.trajectory_latent, self.trajectory_time = self._linspace(self.n_points)
        if cell_type_key is not None:
            self._compute_cell_types(adata, cell_type_key)

    def sample_uniformly(self, n_samples, seed=0):
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(self.trajectory_latent), size=n_samples)
        return self.trajectory_latent[indices], self.trajectory_time[indices]

    def _compute_cell_types(self, adata, cell_type_key):
        knc = KNeighborsClassifier(n_neighbors=50)
        knc.fit(adata.obsm[self.rep_key], adata.obs[cell_type_key])
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
        rotation = rotation @ np.array([[-1, 0], [0, 1]])
    adata.obsm["decipher_v_corrected"] = adata.obsm["decipher_v"] @ rotation

    if flip_latent_z:
        # We want z to be correlated positively with the components
        z_sign_correction = np.sign(
            np.corrcoef(adata.obsm["decipher_z"], y=adata.obsm["decipher_v_corrected"], rowvar=False,)[
                : adata.obsm["decipher_z"].shape[1], adata.obsm["decipher_z"].shape[1] :
            ].sum(axis=1)
        )
    else:
        z_sign_correction = 1.0

    adata.obsm["decipher_z_corrected"] = adata.obsm["decipher_z"] * z_sign_correction
    adata.uns["decipher_rotation"] = rotation


def cluster_representations(adata, leiden_resolution=1.0, seed=0, rep_key="decipher_z_corrected"):
    sc.pp.neighbors(
        adata,
        n_neighbors=10,
        random_state=seed,
        use_rep=rep_key,
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
    start_marker=None,
    end_marker=None,
    subset_column=None,
    subset_value=None,
    trajectory_point_density=50,
    cell_types_key=None,
    return_tree=False,
    rep_key="decipher_v_corrected",
):
    cells_location = pd.DataFrame(adata.obsm[rep_key])
    cells_location["cluster"] = adata.obs["PhenoGraph_clusters"].values
    markers = [start_marker, end_marker]

    for m in markers:
        if m is not None:
            cells_location[m] = adata[:, m].X.toarray().reshape(-1)

    aggregation = {
        0: "mean",
        1: "mean",
        **dict.fromkeys(markers, "mean"),
    }
    if None in aggregation:
        del aggregation[None]
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
    if return_tree:
        return tree
    start = cluster_data[start_marker].sort_values().index[-1]
    end = cluster_data[end_marker].sort_values().index[-1]
    trajectory_nodes = nx.shortest_path(tree, start, end)

    trajectory_nodes_location = np.array(
        [cluster_data.loc[cluster_id, [0, 1]] for cluster_id in trajectory_nodes]
    ).astype(np.float32)

    trajectory_length = 0
    for i in range(1, len(trajectory_nodes_location)):
        trajectory_length += distance_func(trajectory_nodes_location[i - 1], trajectory_nodes_location[i])

    trajectory = Trajectory(trajectory_nodes_location, adata, cell_types_key, trajectory_point_density, rep_key=rep_key)

    uns_key = "decipher_trajectories"
    if uns_key not in adata.uns:
        adata.uns[uns_key] = dict()

    if subset_column is None:
        # all cells were used
        key = "all_cells"
    else:
        key = subset_column + "=" + subset_value

    adata.uns[uns_key][key] = {
        "times": trajectory.trajectory_time,
        "points": trajectory.trajectory_latent,
        "checkpoints": trajectory.checkpoints,
        "clusters": trajectory_nodes,
    }

    return trajectory


def compute_decipher_time(adata, trajectories, n_neighbors=10):
    from sklearn.neighbors import KNeighborsRegressor

    adata.obs["origin_cat"] = adata.obs["origin"].astype("category")
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    Xs = []
    Ts = []
    for origin in trajectories:
        X_tmp = trajectories[origin].trajectory_latent @ np.array([[1, 0, 0], [0, 1, 0]])
        X_tmp[:, 2] = adata.obs["origin_cat"].cat.categories.tolist().index(origin) * 1000
        Xs.append(X_tmp)
        Ts.append(trajectories[origin].trajectory_time)
    X = np.concatenate(Xs)
    t = np.concatenate(Ts)
    knn.fit(X, t)
    Xt = adata.obsm["decipher_v_corrected"] @ np.array([[1, 0, 0], [0, 1, 0]])
    Xt[:, 2] = adata.obs["origin_cat"].cat.codes.values * 1000
    adata.obs["decipher_time"] = knn.predict(Xt)


def compute_decipher_time_new(adata, n_neighbors=10):
    from sklearn.neighbors import KNeighborsRegressor

    adata.obs["decipher_time"] = np.nan
    for name, trajectory in adata.uns["decipher_trajectories"].items():
        knn = KNeighborsRegressor(n_neighbors=n_neighbors)
        knn.fit(trajectory["points"], trajectory["times"])
        is_on_trajectory = adata.obs["PhenoGraph_clusters"].isin(trajectory["clusters"])
        cells_on_trajectory_index = adata.obs[is_on_trajectory].index
        cells_on_trajectory_idx = np.where(is_on_trajectory)[0]

        adata.obs.loc[cells_on_trajectory_index, "decipher_time"] = knn.predict(adata.obsm["decipher_v_corrected"][cells_on_trajectory_idx])

def gene_patterns_from_decipher_trajectory(
    adata,
    model,
    trajectory_name=None,
    l_scale=10_000,
    return_mean=True,
    n_samples=1,
):
    if trajectory_name is None:
        # recursively sample from all available trajectories
        for key in adata.uns["decipher_trajectories"]:
            gene_patterns_from_decipher_trajectory(adata, model, key, l_scale, n_samples, return_mean)
        return

    trajectory_latent = adata.uns["decipher_trajectories"][trajectory_name]["points"]
    trajectory_times = adata.uns["decipher_trajectories"][trajectory_name]["times"]
    if "decipher_rotation" in adata.uns:
        # need to undo rotation of the decipher_v_corrected space
        trajectory_latent = trajectory_latent @ np.linalg.inv(adata.uns["decipher_rotation"])

    trajectory_latent = torch.FloatTensor(trajectory_latent)
    z_mean, z_scale = model.decoder_p_to_z(trajectory_latent)
    z_scale = torch.nn.functional.softplus(z_scale)

    if return_mean:
        samples_z = z_mean
    else:
        samples_z = torch.distributions.Normal(z_mean, z_scale).sample(sample_shape=(n_samples,))

    samples_genes = torch.nn.functional.softmax(model.decoder_z_to_x(samples_z), dim=-1) * l_scale
    samples_genes = samples_genes.detach().numpy()

    if n_samples > 1:
        return samples_genes

    uns_key = "decipher_gene_patterns"
    if uns_key not in adata.uns:
        adata.uns[uns_key] = dict()
    adata.uns[uns_key][trajectory_name] = pd.DataFrame(
        samples_genes,
        index=pd.Series(trajectory_times, name="times"),
        columns=adata.var_names,
    )


def compute_disruption_scores(adata):
    """Compute all possible disruption scores:
    - shape: ||beta[0] - beta[1]||_2
    - scale: | log(s[0]) - log(s[1]) |
    - combined: || log(beta[0]*s[0]) - log(beta[1]*s[1]) ||

    """
    disruptions = []
    for g_id in range(len(adata.var_names)):
        beta_g = adata.uns["decipher_basis_decomposition"]["betas"][:, g_id, :]
        gene_scales = adata.uns["decipher_basis_decomposition"]["scales"][:, g_id]
        shape_disruption = np.linalg.norm(beta_g[0] - beta_g[1], ord=2)
        #         scale_disruption = abs(gene_scales[0, g_id] - gene_scales[1, g_id])
        scale_disruption = abs(np.log(gene_scales[0]) - np.log(gene_scales[1]))
        combined_disruption = abs(
            np.linalg.norm(
                np.log(gene_scales[0] * beta_g[0]) - np.log(gene_scales[1] * beta_g[1]),
                ord=2,
            )
        )
        disruptions.append(
            (
                adata.var_names[g_id],
                shape_disruption,
                scale_disruption,
                combined_disruption,
            )
        )
    disruptions = pd.DataFrame(disruptions, columns=["gene", "shape", "scale", "combined"])

    gene_mean = adata.X.toarray().mean(axis=0)
    gene_std = adata.X.toarray().std(axis=0)
    disruptions["gene_mean"] = gene_mean
    disruptions["gene_std"] = gene_std

    return disruptions.sort_values("combined", ascending=False)


def compute_decipher_gene_correlation(adata, model):
    decipher_v = torch.FloatTensor(adata.obsm["decipher_v"])
    gene_expression_imputed_from_decipher_v = (
        model.decoder_z_to_x(model.decoder_p_to_z(decipher_v)[0]).detach().numpy()
    )
    adata.varm["decipher_v_corrected_gene_covariance"] = np.cov(
        gene_expression_imputed_from_decipher_v,
        y=adata.obsm["decipher_v_corrected"],
        rowvar=False,
    )[: adata.X.shape[1], adata.X.shape[1]:]

    decipher_z = torch.FloatTensor(adata.obsm["decipher_z"])
    gene_expression_imputed_from_decipher_z = (
        model.decoder_z_to_x(decipher_z).detach().numpy()
    )
    adata.varm["decipher_z_corrected_gene_covariance"] = np.cov(
        gene_expression_imputed_from_decipher_z,
        y=adata.obsm["decipher_z_corrected"],
        rowvar=False,
    )[: adata.X.shape[1], adata.X.shape[1]:]


def reconstruct_gene_expression_from_z(model, data):
    if isinstance(data, sc.AnnData):
        z = data.obsm["decipher_z"]
    else:
        z = data

    z = torch.FloatTensor(z)
    squeeze = False
    if len(z.shape) == 1:
        z = z[None, :]
        squeeze = True

    predicted_gene_expression = torch.softmax(model.decoder_z_to_x(z), dim=-1)

    if squeeze:
        predicted_gene_expression = predicted_gene_expression.squeeze(0)

    return predicted_gene_expression


def reconstruct_gene_expression_from_v(model, data):
    if isinstance(data, sc.AnnData):
        v = data.obsm["decipher_v"]
    else:
        v = data

    v = torch.FloatTensor(v)
    squeeze = False
    if len(v.shape) == 1:
        v = v[None, :]
        squeeze = True

    predicted_z = model.decoder_p_to_z(v)[0]

    if squeeze:
        predicted_z = predicted_z.squeeze(0)

    return reconstruct_gene_expression_from_z(model, predicted_z)
