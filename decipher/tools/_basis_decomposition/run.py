import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.infer
import pyro.optim
import torch
from pyro.infer import Predictive, SVI, Trace_ELBO
from tqdm import tqdm

from decipher.tools._basis_decomposition.inference import get_inference_guide
from decipher.tools._basis_decomposition.model import BasisDecomposition
from decipher.tools.utils import EarlyStopping


def compute_basis_decomposition(
    gene_patterns,
    inference_mode,
    n_basis=5,
    lr=1e-3,
    n_iter=10_000,
    beta_prior=1.0,
    seed=1,
    normalized_mode=True,
    times=None,
    plot_every_k_epochs=-1,
):
    pyro.set_rng_seed(seed)
    gene_patterns = torch.FloatTensor(gene_patterns)
    model = BasisDecomposition(
        n_basis,
        n_genes=gene_patterns.shape[1],
        n_conditions=gene_patterns.shape[0],
        beta_prior=beta_prior,
        normalized_mode=normalized_mode,
    )
    guide = get_inference_guide(model, inference_mode)
    adam = pyro.optim.Adam({"lr": lr})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    pyro.clear_param_store()
    num_iterations = n_iter
    if times is None:
        times = torch.FloatTensor(np.linspace(-10, 10, gene_patterns.shape[-1]))
    else:
        # TODO: ensure that the times are in [-5, 5]-ish, transparently to the user
        times = torch.FloatTensor(times)

    gene_patterns_mean = gene_patterns.mean(axis=(0, 2), keepdim=True)
    gene_patterns_raw = gene_patterns
    gene_patterns = gene_patterns_raw / gene_patterns_mean

    losses = []
    early_stopping = EarlyStopping(patience=100)
    pbar = tqdm(range(num_iterations))
    for epoch in pbar:
        # calculate the loss and take a gradient step
        loss = svi.step(times, gene_patterns)
        reconstruction = ((model._last_patterns - gene_patterns) ** 2).mean().item()
        reconstruction_rel = reconstruction / (gene_patterns ** 2).mean()
        pbar.set_description(
            "Loss: %.1f - Relative Error: %.2f%%" % (loss, reconstruction_rel * 100)
        )
        losses.append(loss)
        if early_stopping(loss):
            break

        if plot_every_k_epochs > 0 and epoch % plot_every_k_epochs == 0:
            from IPython.core import display

            basis = model._last_basis.detach().numpy()
            plt.figure(figsize=(5, 2.5))
            _plot_basis(basis)
            display.clear_output(wait=True)
            display.display(plt.gcf())
            plt.close()

    model.return_basis = False
    predictive = Predictive(
        model, guide=guide, num_samples=10, return_sites=("beta", "_RETURN", "obs")
    )
    samples = predictive(times, gene_patterns)
    samples["_RETURN"] *= gene_patterns_mean
    gene_scales = model.gene_scales * gene_patterns_mean.squeeze(-1)
    samples = summary(samples)

    return model, guide, times, samples, gene_scales, losses


def get_basis(model, guide, gene_patterns, times):
    gene_patterns = torch.FloatTensor(gene_patterns)
    times = torch.FloatTensor(times)
    return_basis_value = model.return_basis
    model.return_basis = True
    predictive = Predictive(
        model, guide=guide, num_samples=10, return_sites=("beta", "_RETURN", "obs")
    )
    samples = predictive(times, gene_patterns)
    samples = summary(samples)
    bases = samples["_RETURN"]["mean"].detach().numpy()
    model.return_basis = return_basis_value
    return bases


def _plot_basis(bases, colors=None):
    for i in range(bases.shape[1]):
        plt.plot(
            bases[:, i],
            c=colors[i] if colors is not None else None,
            label="basis %d" % (i + 1),
            linewidth=3,
        )


def summary(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "values": v,
        }
    return site_stats
