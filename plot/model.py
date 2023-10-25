import numpy as np
import scanpy as sc


# def plot_decipher_z(
#     adata, basis="decipher_v_corrected", decipher_z_key_prefix="decipher_z", **kwargs
# ):
#
#     sc.pl.embedding(
#         adata,
#         basis="decipher_v_corrected",
#         color=["z%d" % (i) for i in range(1, 11)],
#         vmax=lambda xs: np.quantile(xs, 0.99),
#         vmin=lambda xs: np.quantile(xs, 0.01),
#         color_map="cool_r",
#         frameon=False,
#         show=False,
#         sort_order=False,
#         return_fig=True,
#     )
