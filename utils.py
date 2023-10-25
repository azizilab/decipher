DECIPHER_GLOBALS = dict()
DECIPHER_GLOBALS["save_folder"] = "./_decipher_models/"


def create_decipher_uns_key(adata):
    if "decipher" not in adata.uns:
        adata.uns["decipher"] = dict()
    if "trajectories" not in adata.uns["decipher"]:
        adata.uns["decipher"]["trajectories"] = dict()
