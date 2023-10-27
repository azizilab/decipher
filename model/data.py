import logging
import os
import randomname
import time

import torch
import torch.utils.data
import torch.nn.functional

from model import Decipher, DecipherConfig
from utils import DECIPHER_GLOBALS, create_decipher_uns_key

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s",
    level=logging.INFO,  # stream=sys.stdout
)


def get_random_name(seed=None):
    name = randomname.generate(
        ["a/algorithms", "a/food", "a/physics"],
        ["a/colors", "a/emotions"],
        ["n/algorithms", "n/food", "a/physics"],
        seed=seed,
    )
    datetime_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    return f"{datetime_str}-{name}"


def save_decipher_model(adata, model):
    create_decipher_uns_key(adata)
    if "run_id" not in adata.uns["decipher"]:
        adata.uns["decipher"]["run_id"] = get_random_name()
        logging.info(f"Saving decipher model with run_id {adata.uns['decipher']['run_id']}.")
    else:
        logging.info("Overwriting existing decipher model.")

    model_run_id = adata.uns["decipher"]["run_id"]
    save_folder = DECIPHER_GLOBALS["save_folder"]
    full_path = os.path.join(save_folder, model_run_id)
    os.makedirs(full_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(full_path, "decipher_model.pt"))
    adata.uns["decipher"]["config"] = model.config.to_dict()


def load_decipher_model(adata):
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


def make_data_loader_from_adata(adata, batch_size=64, context_discrete_keys=None):
    genes = torch.FloatTensor(adata.X.todense())
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
    )
