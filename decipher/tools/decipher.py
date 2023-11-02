import logging
import os

import numpy as np
import pyro
import scanpy as sc
import torch
from matplotlib import pyplot as plt
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import MultiStepLR
from tqdm import tqdm

from decipher.plot.decipher import decipher as plot_decipher_v
from decipher.tools._decipher import Decipher, DecipherConfig
from decipher.tools._decipher.data import (
    decipher_load_model,
    decipher_save_model,
    make_data_loader_from_adata,
)
from decipher.tools.utils import EarlyStopping
from decipher.utils import DECIPHER_GLOBALS, GIFMaker, is_notebook, load_and_show_gif

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s",
    level=logging.INFO,
)


def predictive_log_likelihood(decipher, dataloader):
    """Compute the predictive log likelihood of the decipher model."""
    if type(dataloader) == sc.AnnData:
        dataloader = make_data_loader_from_adata(dataloader, decipher.config.batch_size)

    decipher.eval()
    log_likelihood = 0
    for xc in dataloader:
        guide_trace = poutine.trace(decipher.guide).get_trace(*xc)
        replayed_model = poutine.replay(decipher.model, trace=guide_trace)
        blocked_replayed_model = poutine.block(replayed_model, expose=["x"])
        model_trace = poutine.trace(blocked_replayed_model).get_trace(*xc)
        log_likelihood += model_trace.log_prob_sum().item()
    return log_likelihood / len(dataloader.dataset)


def _make_train_val_split(adata, val_frac, seed):
    n_val = int(val_frac * adata.shape[0])
    cell_idx = np.arange(adata.shape[0])
    np.random.default_rng(seed).shuffle(cell_idx)
    val_idx = cell_idx[-n_val:]
    adata.obs["decipher_split"] = "train"
    adata.obs.loc[adata.obs.index[val_idx], "decipher_split"] = "validation"
    adata.obs["decipher_split"] = adata.obs["decipher_split"].astype("category")
    logging.info(
        "Added `.obs['decipher_split']`: the Decipher train/validation split.\n"
        f" {n_val} cells in validation set."
    )


def decipher_train(
    adata: sc.AnnData,
    decipher_config=DecipherConfig(),
    plot_every_k_epochs=-1,
    plot_kwargs=None,
):
    """Train a decipher model.

    Parameters
    ----------
    adata: sc.AnnData
        The annotated data matrix.
    decipher_config: DecipherConfig, optional
        Configuration for the decipher model.
    plot_every_k_epochs: int, optional
        If > 0, plot the decipher space every `plot_every_k_epoch` epochs.
        Default: -1 (no plots).
    plot_kwargs: dict, optional
        Additional keyword arguments to pass to `dc.pl.decipher`.

    Returns
    -------
    decipher: Decipher
        The trained decipher model.
    val_losses: list of float
        The validation losses at each epoch.
    `adata.obs['decipher_split']`: categorical
        The train/validation split.
    `adata.obsm['decipher_v']`: ndarray
        The decipher v space.
    `adata.obsm['decipher_z']`: ndarray
        The decipher z space.
    """

    if is_notebook() and plot_every_k_epochs == -1:
        plot_every_k_epochs = 5
        logger.info(
            "Plotting decipher space every 5 epochs by default. "
            "Set `plot_every_k_epoch` to -2 to disable."
        )

    pyro.clear_param_store()
    pyro.util.set_rng_seed(decipher_config.seed)

    _make_train_val_split(adata, decipher_config.val_frac, decipher_config.seed)
    train_idx = adata.obs["decipher_split"] == "train"
    val_idx = adata.obs["decipher_split"] == "validation"
    adata_train = adata[train_idx, :]
    adata_val = adata[val_idx, :]

    if plot_kwargs is None:
        plot_kwargs = dict()

    decipher_config.initialize_from_adata(adata_train)

    dataloader_train = make_data_loader_from_adata(adata_train, decipher_config.batch_size)
    dataloader_val = make_data_loader_from_adata(adata_val, decipher_config.batch_size)

    decipher = Decipher(
        config=decipher_config,
    )
    optimizer = pyro.optim.ClippedAdam(
        {
            "lr": decipher_config.learning_rate,
            "weight_decay": 1e-4,
        }
    )
    elbo = Trace_ELBO()
    svi = SVI(decipher.model, decipher.guide, optimizer, elbo)
    gif_maker = GIFMaker(dpi=120)

    # Training loop
    val_losses = []
    early_stopping = EarlyStopping(patience=5)
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

        for xc in dataloader_train:
            loss = svi.step(*xc)
            train_losses.append(loss)

        train_elbo = np.sum(train_losses) / len(dataloader_train.dataset)
        decipher.eval()
        val_nll = -predictive_log_likelihood(decipher, dataloader_val)
        val_losses.append(val_nll)
        pbar.set_description(
            f"Epoch {epoch} | train elbo: {train_elbo:.2f} | val ll:" f" {val_nll:.2f}"
        )
        if early_stopping(val_nll):
            print("Early stopping.")
            break

        if plot_every_k_epochs > 0 and (epoch % plot_every_k_epochs == 0):
            _decipher_to_adata(decipher, adata)
            plot_decipher_v(adata, basis="decipher_v", **plot_kwargs)
            gif_maker.add_image(plt.gcf())
            if is_notebook():
                from IPython.core import display

                display.clear_output(wait=True)
                display.display(plt.gcf())
            else:
                plt.show()
            plt.close()

    if is_notebook():
        from IPython.core import display

        display.clear_output()
        pbar.display()

    _decipher_to_adata(decipher, adata)
    decipher_save_model(adata, decipher)

    model_run_id = adata.uns["decipher"]["run_id"]
    save_folder = DECIPHER_GLOBALS["save_folder"]
    full_path = os.path.join(save_folder, model_run_id, "decipher_training.gif")
    gif_maker.save_gif(full_path)

    if is_notebook():
        load_and_show_gif(full_path)

    plot_decipher_v(adata, basis="decipher_v", **plot_kwargs)

    return decipher, val_losses


def rot(t, u=1):
    if u not in [-1, 1]:
        raise ValueError("u must be -1 or 1")
    return np.array([[np.cos(t), np.sin(t) * u], [-np.sin(t), np.cos(t) * u]])


def decipher_rotate_space(
    adata,
    v1_col=None,
    v1_order=None,
    v2_col=None,
    v2_order=None,
    auto_flip_decipher_z=True,
):
    """Rotate and flip the decipher space v to maximize the correlation of each decipher component
    with provided columns values from `adata.obs` (e.g. pseudo-time, cell state progression, etc.)

    Parameters
    ----------
    adata: sc.AnnData
       The annotated data matrix.
    v1_col: str, optional
        Column name in `adata.obs` to align the first decipher component with.
        If None, only align the second component (or does not align at all if `v2` is also None).
    v1_order: list, optional
        List of values in `adata.obs[v1_col]`. The rotation will attempt to align these values in
        order along the v1 component. Must be provided if `adata.obs[v1_col]` is not numeric.
    v2_col: str, optional
        Column name in `adata.obs` to align the second decipher component with.
        If None, only align the first component (or does not align at all if `v1` is also None).
    v2_order: list, optional
        List of values in `adata.obs[v2_col]`. The rotation will attempt to align these values in
        order along the v2 component. Must be provided if `adata.obs[v2_col]` is not numeric.
    auto_flip_decipher_z: bool, default True
        If True, flip each z to be correlated positively with the components.

    Returns
    -------
    `adata.obsm['decipher_v']`: ndarray
        The decipher v space after rotation.
    `adata.obsm['decipher_z']`: ndarray
        The decipher z space after flipping.
    `adata.uns['decipher']['rotation']`: ndarray
        The rotation matrix used to rotate the decipher v space.
    `adata.obsm['decipher_v_not_rotated']`: ndarray
        The decipher v space before rotation.
    `adata.obsm['decipher_z_not_rotated']`: ndarray
        The decipher z space before flipping.
    """
    decipher = decipher_load_model(adata)
    _decipher_to_adata(decipher, adata)

    def process_col_obs(v_col, v_order):
        if v_col is not None:
            v_obs = adata.obs[v_col]
            if v_order is not None:
                v_obs = v_obs.astype("category").cat.set_categories(v_order)
                v_obs = v_obs.cat.codes.replace(-1, np.nan)
            v_valid_cells = ~v_obs.isna()
            v_obs = v_obs[v_valid_cells].astype(float)
            return v_obs, v_valid_cells
        return None, None

    v1_obs, v1_valid_cells = process_col_obs(v1_col, v1_order)
    v2_obs, v2_valid_cells = process_col_obs(v2_col, v2_order)

    def score_rotation(r):
        rotated_space = adata.obsm["decipher_v"] @ r
        score = 0
        if v1_col is not None:
            score += np.corrcoef(rotated_space[v1_valid_cells, 0], v1_obs)[1, 0]
            score -= np.abs(np.corrcoef(rotated_space[v1_valid_cells, 1], v1_obs)[1, 0])
        if v2_col is not None:
            score += np.corrcoef(rotated_space[v2_valid_cells, 1], v2_obs)[1, 0]
            score -= np.abs(np.corrcoef(rotated_space[v2_valid_cells, 0], v2_obs)[1, 0])
        return score

    if v1_col is not None or v2_col is not None:
        rotation_scores = []
        for t in np.linspace(0, 2 * np.pi, 100):
            for u in [-1, 1]:
                rotation = rot(t, u)
                rotation_scores.append((score_rotation(rotation), rotation))
        best_rotation = max(rotation_scores)[1]

        adata.obsm["decipher_v_not_rotated"] = adata.obsm["decipher_v"].copy()
        adata.obsm["decipher_v"] = adata.obsm["decipher_v"] @ best_rotation
        adata.uns["decipher"]["rotation"] = best_rotation

    if auto_flip_decipher_z:
        # flip each z to be correlated positively with the components
        dim_z = adata.obsm["decipher_z"].shape[1]
        z_v_corr = np.corrcoef(adata.obsm["decipher_z"], y=adata.obsm["decipher_v"], rowvar=False)
        z_sign_correction = np.sign(z_v_corr[:dim_z, dim_z:].sum(axis=1))
        adata.obsm["decipher_z_not_rotated"] = adata.obsm["decipher_z"].copy()
        adata.obsm["decipher_z"] = adata.obsm["decipher_z"] * z_sign_correction


def decipher_gene_imputation(adata):
    """Impute gene expression from the decipher model.

    Parameters
    ----------
    adata: sc.AnnData
        The annotated data matrix.

    Returns
    -------
    `adata.layers['decipher_imputed']`: ndarray
        The imputed gene expression.
    """
    decipher = decipher_load_model(adata)
    imputed = decipher.impute_gene_expression_numpy(adata.X.toarray())
    adata.layers["decipher_imputed"] = imputed
    logging.info("Added `.layers['imputed']`: the Decipher imputed data.")


def _decipher_to_adata(decipher, adata):
    """Compute the decipher v and z spaces from the decipher model. Add them to `adata.obsm`.

    Parameters
    ----------
    decipher: Decipher
        The decipher model.
    adata: sc.AnnData
        The annotated data matrix.

    Returns
    -------
    `adata.obsm['decipher_v']`: ndarray
        The decipher v space.
    `adata.obsm['decipher_z']`: ndarray
        The decipher z space.
    """
    decipher.eval()
    latent_v, latent_z = decipher.compute_v_z_numpy(adata.X.toarray())
    adata.obsm["decipher_v"] = latent_v
    adata.obsm["decipher_z"] = latent_z
    logging.info("Added `.obsm['decipher_v']`: the Decipher v space.")
    logging.info("Added `.obsm['decipher_z']`: the Decipher z space.")
