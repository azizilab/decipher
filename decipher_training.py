import matplotlib.pyplot as plt
import numpy as np
import pyro
import scanpy as sc
import torch
from pyro.infer import Trace_ELBO, SVI
from pyro.optim import MultiStepLR

from decipher.tools._decipher.decipher import make_data_loader_from_adata, Decipher


def plot_decipher(adata, title=None, **kwargs):
    if "color" not in kwargs:
        kwargs["color"] = ["log1p_total_counts"]
    with plt.rc_context({"figure.figsize": (6, 4), "figure.dpi": (120)}):
        adata_log = sc.pp.log1p(adata, copy=True, base=10)
        fig = sc.pl.embedding(
            adata_log,
            basis="decipher_v",
            frameon=False,
            title=title,
            **kwargs,
        )
    return fig


def decipher_to_adata(decipher, adata, dataloader):
    decipher.eval()
    latent_z, latent_v = decipher.guide(*dataloader.dataset.tensors)
    adata.obsm["decipher_v"] = latent_v.detach().numpy()
    adata.obsm["decipher_z"] = latent_z.detach().numpy()


def train_simple(
    adata,
    decipher_config,
    learning_rate,
    n_epochs=61,
    batch_size=64,
    plot_every_k_epoch=-1,
    plot_kwargs=None,
):
    pyro.util.set_rng_seed(decipher_config.seed)
    pyro.clear_param_store()
    pyro.util.set_rng_seed(decipher_config.seed)

    if plot_kwargs is None:
        plot_kwargs = dict()

    num_genes = adata.shape[1]
    minibatch_rescaling = adata.shape[0] / batch_size
    decipher_config.minibatch_rescaling = minibatch_rescaling

    dataloader = make_data_loader_from_adata(adata, batch_size)
    if len(dataloader.dataset.tensors) > 1:
        raise ValueError("Only single dataset supported, context is deprecated.")

    decipher = Decipher(
        genes_dim=num_genes,
        decipher_config=decipher_config,
    )

    scheduler = MultiStepLR(
        {
            "optimizer": torch.optim.Adam,
            "optim_args": {"lr": learning_rate, "weight_decay": 1e-4},
            "milestones": [100],
            "gamma": 0.2,
        }
    )

    elbo = Trace_ELBO()
    svi = SVI(decipher.model, decipher.guide, scheduler, elbo)
    decipher.beta = 1e-1
    # Training loop
    decipher.train()
    for epoch in range(n_epochs):
        losses = []
        for xc in dataloader:
            loss = svi.step(*xc)
            losses.append(loss)
        scheduler.step()
        print("[Epoch %04d]  Loss: %.5f" % (epoch, np.mean(losses)))
        if plot_every_k_epoch > 0 and (epoch % plot_every_k_epoch == 0):
            decipher_to_adata(decipher, adata, dataloader)
            plot_decipher(adata, **plot_kwargs)
            plt.show()

    decipher_to_adata(decipher, adata, dataloader)

    return decipher, adata
