import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from scipy.stats import mannwhitneyu
import anndata
import scanpy as sc 
# from itertools import combinations
# import seaborn as sns

def sample_cells_with_replacement(gene_cell_df, cell_name, sample_size=1000):
    current_cell_type_df = gene_cell_df[cell_name]
    num_columns = current_cell_type_df.shape[1]
    if num_columns < sample_size:
        print(f"{cell_name} has fewer than {sample_size} cells.")
        return current_cell_type_df
    sampled_indices = np.random.choice(range(num_columns), size=sample_size, replace=True)
    sampled_columns_df = current_cell_type_df.iloc[:, sampled_indices]
    return sampled_columns_df

def compute_empirical_covariance(imputed_expression):
    covariance_matrix = np.cov(imputed_expression, rowvar=False)
    gene_names = imputed_expression.columns
    covariance_matrix = pd.DataFrame(covariance_matrix, index=gene_names, columns=gene_names)
    return covariance_matrix

def log_volume_of_nonzero_singular_values(covariance_matrix):
    U, singular_values, V = np.linalg.svd(covariance_matrix, full_matrices=False)
    nonzero_singular_values = singular_values[singular_values > 10**(-30)] #1e-6 #1e-30 #plot range of eigenvalues
    log_volume = np.sum(np.log(nonzero_singular_values))
    total_genes = covariance_matrix.shape[0]
    normalized_log_volume = log_volume / total_genes
    return normalized_log_volume

def phenotypic_volume(adata, layer = None, subset = [], num_iterations = 20):

    if not layer:
        adata.layers["counts"] = adata.X.copy()
        #adata.obs['original_total_counts'] = adata.obs['total_counts']
        adata.obs['original_total_counts'] = adata.obs['nCount_RNA']
        sc.pp.normalize_total(adata, exclude_highly_expressed=True)
        sc.pp.log1p(adata)
        adata.layers["log_counts"] = adata.X.copy()
        df = adata.to_df(layer='log_counts').transpose()
    else:
        df = adata.to_df(layer = 'log_counts').transpose()

    if not subset:
        subset = list(pd.unique(df.columns))

    volumes = {}
    sample_size = 1000
    largest_cluster = float("-inf")
    for cell in subset:
        largest_cluster = max(df[cell].shape[1], largest_cluster)
    for cell in subset:
        current_cell_type_df = df[cell]
        num_columns = current_cell_type_df.shape[1]
        sample_size = min(sample_size, num_columns)
    num_iterations = 10*largest_cluster//sample_size
    for cell in subset:
        current_cell_type_df = df[cell]
        num_columns = current_cell_type_df.shape[1]
        sample_size = min(sample_size, num_columns)
    print(num_iterations)
    for cell in subset:
        cell_type_volumes = []
        for _ in range(num_iterations): #largest cluster/group of cells (n), every cell should have good probability 20*1000/n should be about 5-10ish
            X = sample_cells_with_replacement(df, cell, sample_size)
            Y = compute_empirical_covariance(X)
            Z = log_volume_of_nonzero_singular_values(Y)
            cell_type_volumes.append(Z)
        volumes[cell] = cell_type_volumes

    return pd.DataFrame(volumes)

def main():
    adata_origin = sc.read('data_tum_merged_v5.h5ad')
    adata = adata_origin.copy()
    response_map = {'CR': 'R', 'PR': 'R', 'SD':'NR', 'PD':'NR'}
    for response in response_map:
        adata.obs['best_response'] = adata.obs['best_response'].replace(response, response_map[response])
    adata.obs_names = adata.obs['best_response']
    PV = phenotypic_volume(adata)
    PV.to_csv(f'tumor-response.csv')

    adata_origin = sc.read('data_tum_merged_v5.h5ad')
    adata = adata_origin.copy()
    adata.obs_names = adata.obs['time']
    PV = phenotypic_volume(adata)
    PV.to_csv(f'tumor-time.csv')

    adata_origin = sc.read('data_tum_merged_v5.h5ad')
    adata = adata_origin.copy()
    response_map = {'CR': 'R', 'PR': 'R', 'SD':'NR', 'PD':'NR'}
    for response in response_map:
        adata.obs['best_response'] = adata.obs['best_response'].replace(response, response_map[response])
    concatenated_values = adata.obs["best_response"].astype(str) + '_' + adata.obs["time"].astype(str)
    adata.obs_names = concatenated_values
    PV = phenotypic_volume(adata)
    PV.to_csv(f'tumor-response-plus-time.csv')

if __name__ == "__main__":
    main()



