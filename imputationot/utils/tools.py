import scanpy as sc
import anndata as ad
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, mean_absolute_error, mean_squared_error, confusion_matrix, jaccard_score
from math import sqrt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap


random_seed = 2024
np.random.seed(random_seed)
torch.manual_seed(random_seed)
sc.settings.seed = random_seed
torch.cuda.manual_seed(random_seed)


def visualize_clusters(data, labels, save_path):
    """
    Visualize single-cell clusters and their centers from a high-dimensional PyTorch tensor using UMAP.
    
    Parameters:
        data (torch.Tensor): 2D PyTorch tensor with shape (n_samples, n_features).
        labels (list or torch.Tensor): List or tensor indicating the cluster label for each row in `data`.
        save_path (str): File path to save the plot.
    """
    # Convert data and labels to numpy for easier handling
    data_np = data.detach().cpu().numpy()
    labels_np = np.array(labels)
    
    # Perform UMAP dimensionality reduction to 2D
    reducer = umap.UMAP(n_components=2, random_state=42)
    data_2d = reducer.fit_transform(data_np)
    
    # Calculate cluster centers in the reduced 2D space
    unique_labels = np.unique(labels_np)
    cluster_centers = np.array([data_2d[labels_np == label].mean(axis=0) for label in unique_labels])

    # Plot settings
    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")

    # Scatter plot for each cluster without adding legend labels
    for label in unique_labels:
        cluster_data = data_2d[labels_np == label]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], alpha=0.6, s=5)
    
    # Plot cluster centers and add a single legend entry
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
                color='black', marker='X', s=100, label='Cluster Centers')
    
    # Add titles and labels
    plt.title('Single-cell Cluster Visualization with Cluster Centers')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.legend()  # Only show the cluster centers in the legend

    # Save the plot to the specified path
    plt.savefig(save_path, format='png', dpi=300)
    plt.close()
    print(f"Figure saved to {save_path}")


def correlation_matrix(X):
    X_centered = X - X.mean(dim=0, keepdim=True)
    cov_matrix = X_centered.t() @ X_centered
    variance = cov_matrix.diag().unsqueeze(1)
    correlation = cov_matrix / torch.sqrt(variance @ variance.t())
    return correlation
    

def correlation_matrix_distance(X1, X2):
    corr1 = correlation_matrix(X1)
    corr2 = correlation_matrix(X2)
    
    distance = torch.norm(corr1 - corr2, p='fro')
    return distance


def purity_score(y_true, y_pred):
    contingency_matrix = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def calculate_mae_rmse(imputed, ground_truth, mask):
    imputed_values = imputed[mask].detach().cpu().numpy()
    ground_truth_values = ground_truth[mask].detach().cpu().numpy()
    
    mae = mean_absolute_error(ground_truth_values, imputed_values)
    rmse = sqrt(mean_squared_error(ground_truth_values, imputed_values))
    
    return mae, rmse

    
def cluster_with_kmeans(adata, n_clusters=10, use_pca=True, n_pcs=50, tag='cell_type'):
    data = adata.X

    if use_pca:
        sc.tl.pca(adata, n_comps=n_pcs)
        data = adata.obsm['X_pca']

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    true_labels = adata.obs[tag]
    predicted_labels = kmeans.fit_predict(data).astype(str)
    
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    purity = purity_score(true_labels, predicted_labels)
    jaccard = jaccard_score(true_labels, predicted_labels, average='macro')

    return ari, nmi, purity, jaccard


def cluster_with_leiden(adata, tag='cell_type', resolution_values=[0.10, 0.20, 0.30, 0.40]):
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, use_rep="X_pca")
    true_labels = adata.obs[tag]
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