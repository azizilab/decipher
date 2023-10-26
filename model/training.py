import numpy as np
import pyro
import torch
from matplotlib import pyplot as plt
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import MultiStepLR
from tqdm import tqdm

from model import Decipher, DecipherConfig
from model.data import make_data_loader_from_adata, save_decipher_model

"""
Plan:
sdc.tl.decipher(adata, ...) should :
    - create a Decipher object
    - train it
    - save it
    - rotate the latent space to match the first time point
    - add the latent space to adata.obsm
"""

from plot.model import plot_decipher_v
import scanpy as sc


def fit_decipher(
    adata: sc.AnnData,
    decipher_config=DecipherConfig(),
    plot_every_k_epoch=-1,
):
    pyro.clear_param_store()
    pyro.util.set_rng_seed(decipher_config.seed)

    # # split data into train and validation
    # val_frac = decipher_config.val_frac
    # n_val = int(val_frac * adata.shape[0])
    # cell_idx = np.arange(adata.shape[0])
    # np.random.default_rng(0).shuffle(cell_idx)
    # train_idx = cell_idx[:-n_val]
    # val_idx = cell_idx[-n_val:]
    # adata_train = adata[train_idx, :]
    # adata_val = adata[val_idx, :]

    decipher_config.initialize_from_adata(adata)

    # dataloader_train = make_data_loader_from_adata(adata_train, decipher_config.batch_size)

    # dataloader_val = make_data_loader_from_adata(adata_val, decipher_config.batch_size)
    dataloader = make_data_loader_from_adata(adata, decipher_config.batch_size)
    if len(dataloader.dataset.tensors) > 1:
        raise ValueError("Only single dataset supported, context is deprecated.")

    decipher = Decipher(
        config=decipher_config,
    )

    scheduler = MultiStepLR(
        {
            "optimizer": torch.optim.Adam,
            "optim_args": {"lr": decipher_config.learning_rate, "weight_decay": 1e-4},
            "milestones": [100],
            "gamma": 0.8,
        }
    )

    elbo = Trace_ELBO()
    svi = SVI(decipher.model, decipher.guide, scheduler, elbo)
    # Training loop
    pbar = tqdm(range(decipher_config.n_epochs))
    for epoch in pbar:
        train_losses = []
        decipher.train()
        if epoch > 0:
            # freeze the batch norm layers after the first epoch
            # 1) the batch norm layers helps with the initialization
            # 2) but then, they seem to imply a strong normal prior on the latent space
            for module in decipher.modules():
                if isinstance(module, torch.nn.BatchNorm1d):
                    module.eval()
        for xc in dataloader:
            loss = svi.step(*xc)
            train_losses.append(loss)
        scheduler.step()

        val_losses = train_losses
        # val_losses = []
        # decipher.eval()
        # with torch.no_grad():
        #     for xc in dataloader_val:
        #         loss = svi.evaluate_loss(*xc)
        #         val_losses.append(loss)
        pbar.set_description(
            f"Epoch {epoch} | train elbo: {np.mean(train_losses):.2f} | val elbo:"
            f" {np.mean(val_losses):.2f}"
        )

        if plot_every_k_epoch > 0 and (epoch % plot_every_k_epoch == 0):
            _decipher_to_adata(decipher, adata, dataloader)
            plot_decipher_v(adata, ["cell_type_merged", "log1p_total_counts"], basis="decipher_v")
            plt.show()

    _decipher_to_adata(decipher, adata, dataloader)
    save_decipher_model(adata, decipher)

    return decipher


def _decipher_to_adata(decipher, adata, dataloader):
    decipher.eval()
    latent_z, latent_v = decipher.guide(*dataloader.dataset.tensors)
    adata.obsm["decipher_v"] = latent_v.detach().numpy()
    adata.obsm["decipher_z"] = latent_z.detach().numpy()
