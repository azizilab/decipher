import logging

import numpy as np
import pandas as pd

from decipher.tools._basis_decomposition.inference import InferenceMode
from decipher.tools._basis_decomposition.run import (
    compute_basis_decomposition as run_compute_basis_decomposition,
    get_basis,
)
from decipher.utils import is_notebook

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s",
    level=logging.INFO,  # stream=sys.stdout
)


def basis_decomposition(
    adata,
    pattern_names=None,
    n_basis=5,
    n_iter=10_000,
    lr=5e-3,
    beta_prior=1,
    seed=0,
    plot_every_k_epochs=-1,
):
    """Compute the basis decomposition of gene patterns.

    Parameters
    ----------
    adata : sc.AnnData
        The annotated data matrix.
    pattern_names : list of str, optional
        The names of the gene patterns to use. If None, use all gene patterns available in
        `adata.uns['decipher']['gene_patterns']`.
    n_basis : int, default 5
        The number of basis to use.
    n_iter : int, default 10_000
        The number of iterations to run.
    lr : float, default 5e-3
        The learning rate.
    beta_prior : float, default 1
        The prior on the beta parameter. The lower the value, the more sparse the betas.
    seed : int, default 0
        The random seed to use.
    plot_every_k_epochs : int, default -1, or 100 in jupyter notebook
        Plot the loss every `plot_every_k_epochs` epochs. If <0, do not plot.

    Returns
    -------
    losses : list of float
        The losses at each iteration.
    `adata.uns['decipher']['basis_decomposition']` : dict
        The basis decomposition results.
        - `scales` : np.ndarray (n_patterns, n_genes) - the scales of each gene in each pattern
        - `betas` : np.ndarray (n_patterns, n_genes, n_basis) - the betas of each pattern
        - `times` : np.ndarray (n_times,) - the time points
        - `basis` : np.ndarray (n_times, n_basis) - the basis values at each time point
        - `length` : int - the length of the gene patterns
        - `gene_patterns_reconstruction` : dict - the reconstruction of each gene pattern
        - `pattern_names` : list of str - the names of the gene patterns ordered as in each array
        - `betas_samples` : np.ndarray (n_samples, n_patterns, n_genes, n_basis) - the betas
            of each pattern sampled from the posterior
    `adata.varm['decipher_betas_{pattern_name}']` : np.ndarray
        The betas of the pattern `pattern_name` for each gene.
    """
    if is_notebook() and plot_every_k_epochs == -1:
        plot_every_k_epochs = 100
        logger.info(
            "Plotting the loss every %d epochs. Set `plot_every_k_epochs` to -2 to disable this "
            "behavior." % plot_every_k_epochs
        )
    if pattern_names is None:
        pattern_names = list(adata.uns["decipher"]["gene_patterns"].keys())
    gene_patterns = [
        adata.uns["decipher"]["gene_patterns"][gp_name]["mean"] for gp_name in pattern_names
    ]
    min_len = min([gp.shape[0] for gp in gene_patterns])
    gene_patterns = [gp[:min_len].T for gp in gene_patterns]
    gene_patterns = np.stack(gene_patterns, axis=0)

    # assume all gene patterns have the same times (it is the case for decipher)
    p_name = pattern_names[0]
    gene_patterns_times = adata.uns["decipher"]["gene_patterns"][p_name]["times"][:min_len]

    trajectory_model, guide, times, samples, gene_scales, losses = run_compute_basis_decomposition(
        gene_patterns,
        InferenceMode.GAUSSIAN_BETA_ONLY,
        n_basis=n_basis,
        lr=lr,
        n_iter=n_iter,
        beta_prior=beta_prior,
        seed=seed,
        times=gene_patterns_times,
        plot_every_k_epochs=plot_every_k_epochs,
    )
    gene_scales = gene_scales.detach().numpy()
    basis = get_basis(trajectory_model, guide, gene_patterns, times)
    betas_shape = [*gene_patterns.shape[:2], n_basis]
    betas_mean = samples["beta"]["mean"].detach().numpy().reshape(betas_shape)

    adata.uns["decipher"]["basis_decomposition"] = {
        "scales": gene_scales,
        "betas": betas_mean,
        "betas_samples": samples["beta"]["values"].detach().numpy().reshape([-1] + betas_shape),
        "basis": basis,
        "times": times.detach().numpy(),
        "length": min_len,
        "gene_patterns_reconstruction": {
            gp_name: samples["_RETURN"]["mean"][i].squeeze().detach().numpy()
            for i, gp_name in enumerate(pattern_names)
        },
        "pattern_names": pattern_names,
    }
    for i, p_name in enumerate(pattern_names):
        adata.varm[f"decipher_betas_{p_name}"] = betas_mean[i, :, :]

    return losses


def disruption_scores(adata, pattern_name_a=0, pattern_name_b=1):
    """Compute the disruption scores:
        - shape: ||beta[0] - beta[1]||_2
        - scale: | log(s[0]) - log(s[1]) |
        - combined: || log(beta[0]*s[0]) - log(beta[1]*s[1]) ||

    Parameters
    ----------
    adata : sc.AnnData
        The annotated data matrix.
    pattern_name_a : str or int, default 0
        The name or index of the first pattern.
    pattern_name_b : str or int, default 1
        The name or index of the second pattern.

    Returns
    -------
    `adata.var['decipher_disruption_shape']` : pd.Series
        The shape disruption scores for each gene.
    `adata.var['decipher_disruption_scale']` : pd.Series
        The scale disruption scores for each gene.
    `adata.var['decipher_disruption_combined']` : pd.Series
        The combined disruption scores for each gene.
    `adata.uns['decipher']['disruption_scores']` : pd.DataFrame
        The disruption scores for each gene.
    `adata.uns['decipher']['disruption_scores_samples']` : pd.DataFrame
        The disruption scores for each gene sampled from the posterior.
    """
    if type(pattern_name_a) == str:
        pattern_name_a = adata.uns["decipher"]["basis_decomposition"]["pattern_names"].index(
            pattern_name_a
        )
    if type(pattern_name_b) == str:
        pattern_name_b = adata.uns["decipher"]["basis_decomposition"]["pattern_names"].index(
            pattern_name_b
        )
    idx_a = pattern_name_a
    idx_b = pattern_name_b

    def pairwise_distances(x, y):
        """
        Parameters
        ----------
        x : array_like
            An array of m samples of k variables with d dimensions. Shape: (m, k, d)
        y : array_like
            An array of n samples of k variables with d dimensions. Shape: (n, k, d)
        Returns
        -------
        distance : ndarray
            The matrix of all pairwise distances, shape = (m, n, k).
        """
        x = np.expand_dims(x, axis=1)  # shape: (m, 1, k, d)
        y = np.expand_dims(y, axis=0)  # shape: (1, n, k, d)
        return np.linalg.norm(x - y, ord=2, axis=-1)

    # betas_samples shape: (n_samples, n_patterns, n_genes, n_times)
    beta_a = adata.uns["decipher"]["basis_decomposition"]["betas_samples"][:, idx_a, :, :]
    beta_b = adata.uns["decipher"]["basis_decomposition"]["betas_samples"][:, idx_b, :, :]
    # (gene) scales shape: (n_patterns, n_genes)
    gene_scales_a = adata.uns["decipher"]["basis_decomposition"]["scales"][idx_a, :]
    gene_scales_b = adata.uns["decipher"]["basis_decomposition"]["scales"][idx_b, :]

    n_genes = len(adata.var_names)
    shape_disruption = pairwise_distances(beta_a, beta_b)
    shape_disruption = shape_disruption.reshape(-1, n_genes)
    scale_disruption = np.abs(np.log(gene_scales_a) - np.log(gene_scales_b))
    combined_disruption = pairwise_distances(
        np.log(gene_scales_a[None, :, None] * beta_a),
        np.log(gene_scales_b[None, :, None] * beta_b),
    )
    combined_disruption = combined_disruption.reshape(-1, n_genes)
    disruptions_mean = pd.DataFrame(
        {
            "gene": adata.var_names,
            "shape": shape_disruption.mean(axis=0),
            "scale": scale_disruption,
            "combined": combined_disruption.mean(axis=0),
        }
    ).set_index("gene")

    adata.var["decipher_disruption_shape"] = disruptions_mean["shape"]
    adata.var["decipher_disruption_scale"] = disruptions_mean["scale"]
    adata.var["decipher_disruption_combined"] = disruptions_mean["combined"]
    logger.info("Added `.var['decipher_disruption_shape']`: shape disruption scores")
    logger.info("Added `.var['decipher_disruption_scale']`: scale disruption scores")
    logger.info("Added `.var['decipher_disruption_combined']`: combined disruption scores")

    n_samples = shape_disruption.shape[0]
    disruptions_samples = pd.DataFrame(
        {
            "gene": np.tile(adata.var_names, n_samples),  # repeat the gene names n_samples times
            "shape": shape_disruption.reshape(-1),
            "combined": combined_disruption.reshape(-1),
        }
    )
    adata.uns["decipher"]["disruption_scores_samples"] = disruptions_samples
    adata.uns["decipher"]["disruption_scores"] = disruptions_mean
    logger.info("Added `.uns['decipher']['disruption_scores']`: disruption scores")
    logger.info(
        "Added `.uns['decipher']['disruption_scores_samples']`: disruption scores probabilistic "
        "samples"
    )
