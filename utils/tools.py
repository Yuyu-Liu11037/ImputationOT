import scanpy as sc
import anndata as ad
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def clustering(adata, resolution_values=[0.10, 0.15, 0.20, 0.25]):
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, use_rep="X_pca")
    true_labels = adata.obs["cell_type"]
    best_ari, best_nmi = 0, 0

    for resolution in resolution_values:
        sc.tl.leiden(adata, resolution=resolution, flavor="igraph", n_iterations=2)
        predicted_labels = adata.obs["leiden"]
    
        ari = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        best_ari = max(best_ari, ari)
        best_nmi = max(best_nmi, nmi)

    return best_ari, best_nmi


def gumbel_sinkhorn(X, tau=1.0, n_iter=20, epsilon=1e-6):
    noise = -torch.log(-torch.log(torch.rand_like(X) + epsilon) + epsilon)
    X = (X + noise) / tau
    X = torch.exp(X)
    for _ in range(n_iter):
        X = X / X.sum(dim=1, keepdim=True)
        X = X / X.sum(dim=0, keepdim=True)
    return X


def calculate_cluster_labels(X):
    adata = ad.AnnData(X.detach().cpu().numpy())
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.leiden(adata, resolution=0.2, flavor="igraph", n_iterations=2)
    predicted_labels = adata.obs["leiden"]
    cluster_labels = torch.tensor(predicted_labels.astype(int).values)
    return cluster_labels

def calculate_cluster_centroids(X, cluster_labels):
    centroids = []
    for cluster in cluster_labels.unique():
        cluster_indices = (cluster_labels == cluster).nonzero(as_tuple=True)[0]
        cluster_centroid = X[cluster_indices].mean(dim=0)
        centroids.append(cluster_centroid)
    centroids = torch.stack(centroids)
    return centroids

def preprocess(adata, omics1="GEX", omics2="ADT", target_sum=1e4, layer=None, n_filter1=None):
    """
    :param adata: dataset name
    :param omics1: GEX or ATAC
    :param omics2: ADT or GEX
    :param target_sum: default 1e4
    :param layer: None or counts
    :param n_filter1: 2000 or 4000
    :return:
    """
    ### preprocess
    adata_omics1 = adata[:, adata.var["feature_types"] == omics1].copy()
    adata_omics2 = adata[:, adata.var["feature_types"] == omics2].copy()
    ### step 1
    sc.pp.normalize_total(adata_omics1, target_sum=target_sum, layer=layer)
    sc.pp.normalize_total(adata_omics2, target_sum=target_sum, layer=layer)
    ### step 2
    sc.pp.log1p(adata_omics1, layer=layer)
    sc.pp.log1p(adata_omics2, layer=layer)
    ### step 3
    sc.pp.highly_variable_genes(
        adata_omics1,
        n_top_genes=n_filter1,
        subset=True,
        layer=layer
    )
    if omics2 == "GEX":
        sc.pp.highly_variable_genes(
            adata_omics2,
            subset=True,
            layer=layer
        )
    adata = ad.concat([adata_omics1, adata_omics2], axis=1, merge="first")
    return adata
