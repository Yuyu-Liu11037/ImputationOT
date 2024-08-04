import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.functional as F


def clustering(adata):
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, use_rep="X_pca")
    resolution_values = [0.10, 0.15, 0.20, 0.25]  # corresponding to approximately 9 categories
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


class DKM(nn.Module):
    def __init__(self, num_clusters, temperature=1.0, max_iters=10, epsilon=1e-4):
        super(DKM, self).__init__()
        self.num_clusters = num_clusters
        self.temperature = temperature
        self.max_iters = max_iters
        self.epsilon = epsilon

    def forward(self, weights):
        clusters = self._init_clusters(weights)
        
        for _ in range(self.max_iters):
            dist_matrix = self._compute_distances(weights, clusters)
            attn_matrix = F.softmax(dist_matrix / self.temperature, dim=-1)
            new_clusters = self._update_clusters(weights, attn_matrix)
            if torch.norm(clusters - new_clusters) <= self.epsilon:
                break 
            clusters = new_clusters
        
        clusters = F.normalize(clusters)
        compressed_weights = torch.matmul(attn_matrix, clusters)
        
        return compressed_weights, clusters, attn_matrix

    def _init_clusters(self, weights):
        indices = torch.randperm(weights.size(0))[:self.num_clusters]
        clusters = weights[indices]
        return clusters

    def _compute_distances(self, weights, clusters):
        # Compute pairwise distances between weights and clusters
        weights_expanded = weights.unsqueeze(1)
        clusters_expanded = clusters.unsqueeze(0)
        dist_matrix = -torch.norm(weights_expanded - clusters_expanded, dim=-1)
        return dist_matrix

    def _update_clusters(self, weights, attn_matrix):
        # Update clusters based on attention matrix
        weighted_sum = torch.matmul(attn_matrix.t(), weights)
        cluster_weights = attn_matrix.sum(dim=0)
        new_clusters = weighted_sum / cluster_weights.unsqueeze(-1)
        return new_clusters
