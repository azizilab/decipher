import numpy as np
import pandas as pd

from decipher.tools._basis_decomposition.inference import InferenceMode

from decipher.tools._basis_decomposition.run import (
    compute_basis_decomposition as run_compute_basis_decomposition,
    get_basis,
)


def basis_decomposition(
    adata,
    pattern_names=None,
    n_basis=5,
    n_iter=10_000,
    lr=1e-2,
    beta_prior=1,
    seed=0,
):
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
    )
    gene_scales = gene_scales.detach().numpy()
    betas = samples["beta"]["mean"].squeeze().detach().numpy()
    basis = get_basis(trajectory_model, guide, gene_patterns, times)
    adata.uns["decipher"]["_basis_decomposition"] = {
        "scales": gene_scales,
        "betas": betas,
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
        adata.varm[f"decipher_betas_{p_name}"] = betas[i, :, :]

    return losses


def disruption_scores(adata):
    """Compute all possible disruption scores:
    - shape: ||beta[0] - beta[1]||_2
    - scale: | log(s[0]) - log(s[1]) |
    - combined: || log(beta[0]*s[0]) - log(beta[1]*s[1]) ||

    """
    disruptions = []
    for g_id in range(len(adata.var_names)):
        beta_g = adata.uns["decipher"]["_basis_decomposition"]["betas"][:, g_id, :]
        gene_scales = adata.uns["decipher"]["_basis_decomposition"]["scales"][:, g_id]
        shape_disruption = np.linalg.norm(beta_g[0] - beta_g[1], ord=2)
        scale_disruption = abs(np.log(gene_scales[0]) - np.log(gene_scales[1]))
        combined_disruption = abs(
            np.linalg.norm(
                np.log(gene_scales[0] * beta_g[0]) - np.log(gene_scales[1] * beta_g[1]), ord=2
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
    disruptions = pd.DataFrame(
        disruptions, columns=["gene", "shape", "scale", "combined"]
    ).set_index("gene")
    adata.var["decipher_disruption_shape"] = disruptions["shape"]
    adata.var["decipher_disruption_scale"] = disruptions["scale"]
    adata.var["decipher_disruption_combined"] = disruptions["combined"]
    return disruptions
