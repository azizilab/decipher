import logging
import os

import numpy as np
import pyro
import scanpy as sc
import torch
from matplotlib import pyplot as plt
import pyro.optim
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from tqdm import tqdm

from decipher.plot.decipher import decipher as plot_decipher_v
from decipher.tools._decipher import Decipher, DecipherConfig
from decipher.tools._decipher.data import (
    decipher_load_model,
    decipher_save_model,
    make_data_loader_from_adata,
    get_dense_X,
)
from decipher.tools.utils import EarlyStopping
from decipher.utils import DECIPHER_GLOBALS, GIFMaker, is_notebook, load_and_show_gif

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s",
    level=logging.INFO,
)


def predictive_log_likelihood(decipher, dataloader, n_samples=5):
    log_weights = []
    old_beta = decipher.config.beta
    decipher.config.beta = 1.0
    try:
        for i in range(n_samples):
            total_log_prob = 0
            for xc in dataloader:
                xc = [x.to(decipher.device) for x in xc]
                guide_trace = poutine.trace(decipher.guide).get_trace(*xc)
                model_trace = poutine.trace(
                    poutine.replay(decipher.model, trace=guide_trace)
                ).get_trace(*xc)
                total_log_prob += model_trace.log_prob_sum() - guide_trace.log_prob_sum()
            log_weights.append(total_log_prob)

    finally:
        decipher.config.beta = old_beta

    log_z = torch.logsumexp(torch.tensor(log_weights) - np.log(n_samples), 0)
    return log_z.item()


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
    device="cpu",
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
    device: str, optional
        The device to use for training. Default: "cpu".

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

    dataloader_train = make_data_loader_from_adata(
        adata_train, decipher_config.batch_size, drop_last=True
    )
    dataloader_val = make_data_loader_from_adata(adata_val, decipher_config.batch_size)

    decipher = Decipher(
        config=decipher_config,
    )
    decipher.to(device)

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
    if (
        decipher_config.early_stopping_patience is not None
        and decipher_config.early_stopping_patience > 0
    ):
        early_stopping = EarlyStopping(patience=decipher_config.early_stopping_patience)
    else:
        early_stopping = EarlyStopping(patience=int(1e30))

    pbar = tqdm(range(decipher_config.n_epochs))
    last_train_elbo = np.nan
    val_nll = np.nan
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

        n_batches = len(dataloader_train)
        train_elbo = 0
        train_elbo_n_obs = 0
        for xc in dataloader_train:
            try:
                xc = [x.to(device) for x in xc]
                loss = svi.step(*xc)
            except Exception as e:
                print("ERROR", e)
                return decipher, val_losses
            train_losses.append(loss)
            train_elbo += loss
            train_elbo_n_obs += xc[0].shape[0]
            pbar.set_description(
                f"Epoch {epoch} (batch {len(train_losses)}/{n_batches}) | "
                f"| train elbo: {train_elbo / train_elbo_n_obs:.2f} (last epoch: {last_train_elbo:.2f}) | val ll:"
                f" {val_nll:.2f}"
            )

        decipher.eval()
        val_nll = (
            -predictive_log_likelihood(decipher, dataloader_val, n_samples=5) / adata_val.shape[0]
        )
        val_losses.append(val_nll)
        pbar.set_description(
            f"Epoch {epoch} (batch {len(train_losses)}/{n_batches}) | "
            f"| train elbo: {train_elbo / train_elbo_n_obs:.2f} (last epoch: {last_train_elbo:.2f}) | val ll:"
            f" {val_nll:.2f}"
        )
        last_train_elbo = train_elbo / train_elbo_n_obs
        if early_stopping(val_nll):
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

    if early_stopping.has_stopped():
        logger.info("Early stopping has been triggered.")
    _decipher_to_adata(decipher, adata)
    decipher_save_model(adata, decipher)

    model_run_id = adata.uns["decipher"]["run_id"]
    save_folder = DECIPHER_GLOBALS["save_folder"]
    full_path = os.path.join(save_folder, str(model_run_id), "decipher_training.gif")
    if gif_maker.images:
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
            for u in [1, -1]:
                rotation = rot(t, u)
                rotation_scores.append((score_rotation(rotation), rotation))
        best_rotation = max(rotation_scores, key=lambda x: x[0])[1]

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


def decipher_and_gene_covariance(adata):
    if "decipher_imputed" not in adata.layers:
        decipher_gene_imputation(adata)
    gene_expression_imputed = adata.layers["decipher_imputed"]
    adata.varm["decipher_v_gene_covariance"] = np.cov(
        gene_expression_imputed,
        y=adata.obsm["decipher_v"],
        rowvar=False,
    )[: adata.X.shape[1], adata.X.shape[1] :]
    logging.info(
        "Added `.varm['decipher_v_gene_covariance']`: the covariance between Decipher v and each gene."
    )
    adata.varm["decipher_z_gene_covariance"] = np.cov(
        gene_expression_imputed,
        y=adata.obsm["decipher_z"],
        rowvar=False,
    )[: adata.X.shape[1], adata.X.shape[1] :]
    logging.info(
        "Added `.varm['decipher_z_gene_covariance']`: the covariance between Decipher z and each gene."
    )


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
    latent_v, latent_z = decipher.compute_v_z_numpy(get_dense_X(adata))
    adata.obsm["decipher_v"] = latent_v
    adata.obsm["decipher_z"] = latent_z
    logging.info("Added `.obsm['decipher_v']`: the Decipher v space.")
    logging.info("Added `.obsm['decipher_z']`: the Decipher z space.")
