import numpy as np

from basis_decomposition.inference import InferenceMode

from basis_decomposition.run import (
    compute_basis_decomposition as run_compute_basis_decomposition,
    get_basis,
)


def basis_decomposition(
    adata,
    gene_patterns_names=None,
    n_basis=5,
    n_iter=10_000,
    lr=1e-2,
    beta_prior=1,
    seed=0,
):
    if gene_patterns_names is None:
        gene_patterns_names = list(adata.uns["decipher"]["gene_patterns"].keys())
    gene_patterns = [
        adata.uns["decipher"]["gene_patterns"][gp_name]["mean"] for gp_name in gene_patterns_names
    ]
    min_len = min([gp.shape[0] for gp in gene_patterns])
    gene_patterns = [gp[:min_len].T for gp in gene_patterns]
    gene_patterns = np.stack(gene_patterns, axis=0)

    # assume all gene patterns have the same times (it is the case for decipher)
    gp_name = gene_patterns_names[0]
    gene_patterns_times = adata.uns["decipher"]["gene_patterns"][gp_name]["times"][:min_len]

    trajectory_model, guide, times, samples, gene_scales = run_compute_basis_decomposition(
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
    adata.uns["decipher"]["basis_decomposition"] = {
        "scales": gene_scales,
        "betas": betas,
        "basis": basis,
        "times": times.detach().numpy(),
        "length": min_len,
        "gene_patterns_reconstruction": {
            gp_name: samples["_RETURN"]["mean"][i].squeeze().detach().numpy()
            for i, gp_name in enumerate(gene_patterns_names)
        },
    }
