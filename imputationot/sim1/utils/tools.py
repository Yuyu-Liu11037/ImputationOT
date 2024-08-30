import scanpy as sc
import anndata as ad
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, mean_absolute_error, mean_squared_error, confusion_matrix, jaccard_score
from math import sqrt


random_seed = 2024
np.random.seed(random_seed)
torch.manual_seed(random_seed)
sc.settings.seed = random_seed
torch.cuda.manual_seed(random_seed)


def purity_score(y_true, y_pred):
    contingency_matrix = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def calculate_mae_rmse(imputed, ground_truth, mask):
    imputed_values = imputed[mask].detach().cpu().numpy()
    ground_truth_values = ground_truth[mask].detach().cpu().numpy()
    
    mae = mean_absolute_error(ground_truth_values, imputed_values)
    rmse = sqrt(mean_squared_error(ground_truth_values, imputed_values))
    
    return mae, rmse

    
def cluster_with_kmeans(adata, n_clusters=10, use_pca=True, n_pcs=50):
    data = adata.X

    if use_pca:
        sc.tl.pca(adata, n_comps=n_pcs)
        data = adata.obsm['X_pca']

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    true_labels = adata.obs["Group"]
    predicted_labels = kmeans.fit_predict(data).astype(str)
    
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    purity = purity_score(true_labels, predicted_labels)
    jaccard = jaccard_score(true_labels, predicted_labels, average='macro')

    return ari, nmi, purity, jaccard


def cluster_with_leiden(adata, resolution_values=[0.10, 0.20, 0.30, 0.40]):
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, use_rep="X_pca")
    true_labels = adata.obs["Group"]
    best_ari, best_nmi, best_purity, best_jaccard = 0, 0, 0, 0

    for resolution in resolution_values:
        sc.tl.leiden(adata, resolution=resolution, flavor="igraph", n_iterations=2)
        predicted_labels = adata.obs["leiden"]
    
        ari = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        purity = purity_score(true_labels, predicted_labels)
        jaccard = jaccard_score(true_labels, predicted_labels, average='macro')
        length = adata.obs["leiden"].nunique()
        print(f"{resolution}, {length}, {ari:.4f}, {nmi:.4f}, {purity:.4f}")
        best_ari = max(best_ari, ari)
        best_nmi = max(best_nmi, nmi)
        best_purity = max(best_purity, purity)
        best_jaccard = max(best_jaccard, jaccard)
    print()

    return best_ari, best_nmi, best_purity, best_jaccard


def gumbel_sinkhorn(X, tau=1.0, n_iter=20, epsilon=1e-6):
    noise = -torch.log(-torch.log(torch.rand_like(X) + epsilon) + epsilon)
    X = (X + noise) / tau
    X = torch.exp(X)
    for _ in range(n_iter):
        X = X / X.sum(dim=1, keepdim=True)
        X = X / X.sum(dim=0, keepdim=True)
    return X


def calculate_cluster_labels(X, resolution=0.65):
    adata = ad.AnnData(X.detach().cpu().numpy())
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.leiden(adata, resolution=resolution, flavor="igraph", n_iterations=2)
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