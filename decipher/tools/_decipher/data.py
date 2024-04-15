import logging
import os
import time

import numpy as np
import randomname
import torch
import torch.distributions
import torch.nn.functional
import torch.utils.data

from decipher.tools._decipher import Decipher, DecipherConfig
from decipher.utils import DECIPHER_GLOBALS, create_decipher_uns_key

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s",
    level=logging.INFO,
)


def get_dense_X(adata):
    if isinstance(adata.X, np.ndarray):
        return adata.X
    else:
        return adata.X.toarray()


def get_random_name(seed=None):
    name = randomname.generate(
        ["a/algorithms", "a/food", "a/physics"],
        ["a/colors", "a/emotions"],
        ["n/algorithms", "n/food", "a/physics"],
        seed=seed,
    )
    datetime_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    return f"{datetime_str}-{name}"


def decipher_save_model(adata, model, overwrite=False):
    create_decipher_uns_key(adata)

    if "run_id_history" not in adata.uns["decipher"]:
        adata.uns["decipher"]["run_id_history"] = []

    if "run_id" not in adata.uns["decipher"] or not overwrite:
        adata.uns["decipher"]["run_id"] = get_random_name()
        adata.uns["decipher"]["run_id_history"].append(adata.uns["decipher"]["run_id"])
        logging.info(f"Saving decipher model with run_id {adata.uns['decipher']['run_id']}.")
    else:
        logging.info("Overwriting existing decipher model.")

    model_run_id = adata.uns["decipher"]["run_id"]
    save_folder = DECIPHER_GLOBALS["save_folder"]
    full_path = os.path.join(save_folder, model_run_id)
    os.makedirs(full_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(full_path, "decipher_model.pt"))
    adata.uns["decipher"]["config"] = model.config.to_dict()


def decipher_load_model(adata):
    """Load a decipher model whose name is stored in the given AnnData.

    `adata.uns["decipher"]["run_id"]` must be set to the name of the decipher model to load.

    Parameters
    ----------
    adata : sc.AnnData
        The annotated data matrix.

    Returns
    -------
    model : Decipher
        The decipher model.
    """
    create_decipher_uns_key(adata)
    if "run_id" not in adata.uns["decipher"]:
        raise ValueError("No decipher model has been saved for this AnnData object.")

    model_config = DecipherConfig(**adata.uns["decipher"]["config"])
    model = Decipher(model_config)
    model_run_id = adata.uns["decipher"]["run_id"]
    save_folder = DECIPHER_GLOBALS["save_folder"]
    full_path = os.path.join(save_folder, model_run_id)
    model.load_state_dict(torch.load(os.path.join(full_path, "decipher_model.pt")))
    model.eval()
    return model


def make_data_loader_from_adata(adata, batch_size=64, context_discrete_keys=None, **kwargs):
    """Create a PyTorch DataLoader from an AnnData object."""
    genes = torch.FloatTensor(get_dense_X(adata))
    params = [genes]
    context_tensors = []
    if context_discrete_keys is None:
        context_discrete_keys = []

    for key in context_discrete_keys:
        t = torch.IntTensor(adata.obs[key].astype("category").cat.codes.values).long()
        encoded = torch.nn.functional.one_hot(t).float()
        context_tensors.append(encoded)

    if context_tensors:
        context = torch.cat(context_tensors, dim=-1)
        params.append(context)

    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*params),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )
