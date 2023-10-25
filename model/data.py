import dataclasses
import os
import randomname
import time

import torch

from model import Decipher, DecipherConfig
from utils import DECIPHER_GLOBALS, create_decipher_uns_key


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
    if "model_run_id" not in adata.uns["decipher"]:
        adata.uns["decipher"]["model_run_id"] = get_random_name()

    model_run_id = adata.uns["decipher"]["model_run_id"]
    save_folder = DECIPHER_GLOBALS["save_folder"]
    full_path = os.path.join(save_folder, model_run_id)
    os.makedirs(full_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(full_path, "decipher_model.pt"))
    adata.uns["decipher"]["model_config"] = dataclasses.asdict(model.decipher_config)


def load_decipher_model(adata):
    create_decipher_uns_key(adata)
    if "model_run_id" not in adata.uns["decipher"]:
        raise ValueError("No decipher model has been saved for this AnnData object.")

    model_config = DecipherConfig(**adata.uns["decipher"]["model_config"])
    model = Decipher(adata.shape[1], model_config)
    model_run_id = adata.uns["decipher"]["model_run_id"]
    save_folder = DECIPHER_GLOBALS["save_folder"]
    full_path = os.path.join(save_folder, model_run_id)
    model.load_state_dict(torch.load(os.path.join(full_path, "decipher_model.pt")))
    model.eval()
    return model
