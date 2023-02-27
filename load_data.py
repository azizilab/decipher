import scanpy as sc
import scanpy
import numpy as np
import pandas as pd


def load_datasets(
    min_cells=100,
    min_counts=500,
    target_sum=10_000,
    subsample=None,
    seed=0,
    datasets_to_load=None,
):
    """
    subsample: dict[dataset_name: target cells]
    target_sum : if <=0 then doesnt normalize
    """
    import warnings

    warnings.filterwarnings("ignore")
    dataset_paths = {
        "healthy": "data/BM_2019_01_10.h5",
        "p89": "data/08H089_allsorted_cleaned.h5",
        "tet2_p1": "data/tet2/16H008_allsorted_cleaned.h5",
        "tet2_p2": "data/tet2/10H072_allsorted_cleaned.h5",
        "tet2_p3": "data/tet2/AML7_allsorted_cleaned.h5",
        "dnm_07H009": "data/dnm/DNMT3A _cohort_07H009_unsorted_cohortlevel_annotations.h5",
        "dnm_09H058": "data/dnm/DNMT3A _cohort_09H058_unsorted_cohortlevel_annotations.h5",
        "dnm_10H056": "data/dnm/DNMT3A _cohort_10H056_unsorted_cohortlevel_annotations.h5",
        "dnm_11H240": "data/dnm/DNMT3A _cohort_11H240_unsorted_cohortlevel_annotations.h5",
        "dnm_12H007": "data/dnm/DNMT3A _cohort_12H007_unsorted_cohortlevel_annotations.h5",
        # "dnm_09H058_sorted": 'data/dnm/DNMT3A _cohort_09H058_cohortlevel_annotations.h5',
        # "dnm_10H056_sorted": 'data/dnm/DNMT3A _cohort_10H056_cohortlevel_annotations.h5',
        # "dnm_11H240_sorted": 'data/dnm/DNMT3A _cohort_11H240_cohortlevel_annotations.h5',
        # "dnm_12H007_sorted": 'data/dnm/DNMT3A _cohort_12H007_cohortlevel_annotations.h5',
        # "dnm_12H010_sorted": 'data/dnm/DNMT3A _cohort_12H010_cohortlevel_annotations.h5',
        "dnm_09H058_sorted": "data/DNM_sorted_patients/09H058_cohortlevel_annotations.h5",
        "dnm_10H056_sorted": "data/DNM_sorted_patients/10H056_cohortlevel_annotations.h5",
        "dnm_11H240_sorted": "data/DNM_sorted_patients/11H240_cohortlevel_annotations.h5",
        "dnm_12H007_sorted": "data/DNM_sorted_patients/12H007_cohortlevel_annotations.h5",
        "dnm_12H010_sorted": "data/DNM_sorted_patients/12H010_cohortlevel_annotations.h5",
    }
    cell_type_column = {
        "healthy": "cell_type",
        "p89": "blast_cell_type_08H089",
        "tet2_p1": "blast_cell_type_16H008",
        "tet2_p2": "blast_cell_type_10H072",
        "tet2_p3": "blast_cell_type_AML7",
        "dnm_07H009": "types",
        "dnm_10H056": "types",
        "dnm_09H058_sorted": "types",
        "dnm_12H007_sorted": "types",
        "dnm_12H010_sorted": "types",
        "dnm_12H007": "types",
        "dnm_11H240_sorted": "types",
        "dnm_09H058": "types",
        "dnm_11H240": "types",
        "dnm_10H056_sorted": "types",
    }
    if datasets_to_load is not None:
        dataset_paths = {d: dataset_paths[d] for d in datasets_to_load}
    datasets = dict()
    for key, path in dataset_paths.items():
        datasets[key] = scanpy.read(path)

    OBS = [
        "cell_type",
        "NPM1_mut",
        "DNMT3A_mut",
        "DNMT3A_wt",
        "NPM1_wt",
        "phase",
    ]

    np.random.seed(seed)

    gene_intersection = None
    for name, data in datasets.items():
        np.random.seed(seed)
        data.obs["origin"] = name
        if name not in cell_type_column:
            col = "cell_type"
            print(
                "Cell type column for dataset",
                name,
                "not provided",
                "defaulting to `cell_type`",
            )
        else:
            col = cell_type_column[name]
        data.obs["cell_type_merged"] = data.obs[col]

        for c in OBS:
            if c not in data.obs:
                data.obs[c] = "undefined"
        data.X = data.X.astype(float)

        scanpy.pp.filter_genes(data, min_cells=min_cells)
        scanpy.pp.filter_genes(data, min_counts=min_counts)

        if gene_intersection is None:
            gene_intersection = set(data.var_names)
        else:
            gene_intersection.intersection_update(data.var_names)
        print(name, "PREPROCESSED: OK")

    print("align gene names")

    # genes_interest = np.loadtxt('genes.txt', dtype=str)
    # gene_intersection = gene_intersection.intersection(set(genes_interest))

    print("Removing genes starting with RPL*, RPS*, MT-*")
    gene_intersection = filter(lambda x: x[:3] not in ["RPL", "RPS", "MT-"], gene_intersection)
    gene_intersection = list(sorted(gene_intersection))

    if subsample is not None:
        print(
            "Subsampling datasets to: ",
            ", ".join(["%s = %d cells" % (n, subsample[n]) for n in set(subsample).intersection(datasets)]),
        )
        for name, data in datasets.items():
            datasets[name] = data[:, gene_intersection]
            if name in subsample:
                scanpy.pp.subsample(
                    datasets[name], n_obs=min(subsample[name], datasets[name].n_obs), random_state=seed
                )

    # Combine the dataset
    if target_sum is not None and target_sum > 0:
        for name, data in list(datasets.items()):
            scanpy.pp.normalize_total(data, target_sum=target_sum)

    if "healthy" in datasets:
        print("Build combined datasets with healthy")
        for name, data in list(datasets.items()):
            if name == "healthy":
                continue
            datasets[name + "_healthy"] = data.concatenate(datasets["healthy"], batch_key="sample")

    for data in datasets.values():
        data.X = (data.X.astype(np.float64)).astype(np.int64).astype(np.float64)

    return datasets


palette_original = {
    "immature": "#8c564b",
    "blast0": "#1f77b4",
    "blast1": "#ff7f0e",
    "blast2": "#2ca02c",
    "blast3": "#d62728",
    "ery": "#9467bd",
    "lympho": "#e377c2",
    "mep": "#bcbd22",
    "N/A": "#DEDEDE",
    "immature_myelo": "#17becf",  # #DC9E6E
    "healthy": "#1f77b4",
    "p89": "#ff7f0e",
    "tet2_p1": "#ff7f0e",
    "tet2_p2": "#ff7f0e",
    "tet2_p3": "#ff7f0e",
    "AML1": "#9357e2",
    "AML2": "#9357e2",
    "AML3": "#9357e2",
    "Healthy": "#a6e257",
}

palette = {
    "immature": "#b96d40",
    "blast0": "#ffcf9c",
    "blast1": "#4BC4C6",
    "blast2": "#622450",
    "blast3": "#ca054d",
    "ery": "#5b6c5d",
    "lympho": "#FAE500",
    "mep": "#c472d9",
    "N/A": "#DEDEDE",
    "immature_myelo": "#96A360",  # #DC9E6E
    "healthy": "#3ECD51",
    "p89": "#D44E4E",
    "tet2_p1": "#D44E4E",
    "tet2_p2": "#D44E4E",
    "tet2_p3": "#D44E4E",
    "dnm_07H009": "#D44E4E",
    "dnm_10H056": "#D44E4E",
    "dnm_09H058_sorted": "#D44E4E",
    "dnm_12H007_sorted": "#D44E4E",
    "dnm_12H010": "#D44E4E",
    "dnm_12H007": "#D44E4E",
    "dnm_11H240_sorted": "#D44E4E",
    "dnm_09H058": "#D44E4E",
    "dnm_11H240": "#D44E4E",
    "dnm_10H056_sorted": "#D44E4E",
}


def t_test_deg(adata, n_genes_per_cluster, seed=0):
    # pca of n_components
    scanpy.pp.pca(adata, n_comps=50, random_state=seed, use_highly_variable=False)
    # communities, graph, Q = scanpy.external.tl.phenograph(adata.obsm["X_pca"], k=30, seed=seed, n_jobs=1)
    # adata.obs["PhenoGraph_clusters"] = pd.Categorical(communities)
    # adata.uns["PhenoGraph_Q"] = Q
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40, random_state=seed)
    sc.tl.leiden(adata, random_state=seed)
    adata.obs["PhenoGraph_clusters"] = pd.Categorical(adata.obs["leiden"])

    adata_log = scanpy.pp.log1p(adata, copy=True)
    scanpy.tl.rank_genes_groups(
        adata_log,
        "PhenoGraph_clusters",
        method="t-test",
        key_added="t-test",
    )

    ttest = pd.DataFrame()
    for i in np.unique(adata.obs["PhenoGraph_clusters"]):
        ttest_ii = scanpy.get.rank_genes_groups_df(adata_log, key="t-test", group=str(i))["names"]
        gene_list = ttest_ii.values[:n_genes_per_cluster]
        ttest["Cluster " + str(i)] = gene_list
    return ttest


MYELOID = ["CD14", "CD68", "APOE", "CD81", "CD9", "CD163", "CD1C", "FLT3", "CD33"]
NK = ["NCAM1", "GNLY", "NCR1", "KIR2DL4", "KIR2DL3", "KIR3DL2", "KIR2DL1", "KIR3DL1", "KIR3DX1", "KIR3DL2"]
MACROPHAGE = ["CD68", "MRC1", "MSR1", "NRP1", "CD82", "CD14", "CD163", "CD86", "CD81", "C5AR1"]
MONOCYTES = ["CD33", "LYZ", "MPO", "FCN1", "CSF3R", "VCAN"]
B_CELLS = ["CD19", "CD40LG", "IGHD", "EBF1", "TCF3", "CD1C", "MS4A1", "CR2", "CD27", "CD22", "CD79A", "CD79B"]
B_CELLS.extend(["POU2F2", "IGHG1", "IGHM", "EBF1", "PAX5"])
HSC = ["CD34"]
T_CELLS = ["CD3D", "CD3E"]
PROLI_MARKERS = ["MKI67", "PCNA"]
ERYTHROID = ["HBB", "HBD", "HBM", "HBA1", "HBA2", "ALAS2", "SOX6", "GATA1"]
OTHERS = ["AVP", "PROM1", "NPM1", "RUNX1", "SCLT1", "LMO2", "BMI1", "GATA2", "NOTCH2"]
OTHERS.extend(["SLFN13", "TNFSF10", "HOXA9", "MEIS1", "HHEX", "CXCL3", "EEF2", "CD37", "BTF3", "EIF3F"])
MARKER_GENES = sorted(
    set().union(MYELOID, NK, MACROPHAGE, MONOCYTES, B_CELLS, HSC, T_CELLS, PROLI_MARKERS, ERYTHROID, OTHERS)
)

MARKER_GENES_DICT = {
    "myeloid": MYELOID,
    "nk": NK,
    "macrophage": MACROPHAGE,
    "monocytes": MONOCYTES,
    "b_cells": B_CELLS,
    "hsc": HSC,
    "t_cells": T_CELLS,
    "proli_markers": PROLI_MARKERS,
    "erythroid": ERYTHROID,
    "others": OTHERS,
}
