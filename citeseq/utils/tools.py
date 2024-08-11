import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    return best_ari, best_nmi, predicted_labels


def gumbel_sinkhorn(X, tau=1.0, n_iter=20, epsilon=1e-6):
    noise = -torch.log(-torch.log(torch.rand_like(X) + epsilon) + epsilon)
    X = (X + noise) / tau
    X = torch.exp(X)
    for _ in range(n_iter):
        X = X / X.sum(dim=1, keepdim=True)
        X = X / X.sum(dim=0, keepdim=True)
    return X


# def dkm_clustering(W, k, temperature=1.0, n_iter=100, epsilon=1e-4):
#     m, _ = W.shape
#     C = W[torch.randperm(m)[:k]]  # Initialize cluster centers
#     for _ in range(n_iter):
#         D = torch.cdist(W, C)
#         A = F.softmax(-D / temperature, dim=1)
#         C_new = (A.T @ W) / A.sum(dim=0)[:, None]
#         if torch.norm(C_new - C) < epsilon:
#             break
#         C = C_new
#     return C


class DKM(nn.Module):
    """
    DKM clustering module.

    Returns:
        attn_matrix: shape [cell, n_classes], a[i][j] := probability that cell i belongs to class j.
    """
    def __init__(self, num_clusters, temperature=1.0, max_iters=10, epsilon=1e-4):
        super(DKM, self).__init__()
        self.num_clusters = num_clusters
        self.temperature = temperature
        self.max_iters = max_iters
        self.epsilon = epsilon

    def forward(self, weights):
        clusters = self._init_clusters(weights)
        
        for _ in range(self.max_iters):
            dist_matrix = -torch.cdist(weights, clusters)
            attn_matrix = F.softmax(dist_matrix / self.temperature, dim=-1)
            new_clusters = torch.matmul(attn_matrix.t(), weights) / attn_matrix.sum(dim=0).unsqueeze(-1)
            if torch.norm(clusters - new_clusters) <= self.epsilon:
                break
            clusters = new_clusters
        
        compressed_weights = torch.matmul(attn_matrix, clusters)
        return compressed_weights, clusters, attn_matrix

    def _init_clusters(self, weights):
        # k-means++ initialization
        clusters = []
        clusters.append(weights[torch.randint(0, weights.size(0), (1,))].squeeze(0))

        for _ in range(1, self.num_clusters):
            dist_matrix = torch.stack([torch.norm(weights - cluster, dim=1) for cluster in clusters])
            min_dist, _ = torch.min(dist_matrix, dim=0)
            probs = min_dist / min_dist.sum()
            new_cluster = weights[torch.multinomial(probs, 1)]
            clusters.append(new_cluster.squeeze(0))

        clusters = torch.stack(clusters).to(weights.device)
        return clusters
        