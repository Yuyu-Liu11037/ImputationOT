from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score
from sklearn.decomposition import PCA
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from geomloss import SamplesLoss
from scipy.stats import pearsonr
import sys
from sklearn.decomposition import PCA


def visualize_clusters_with_centers(data, labels, save_path):
    """
    Visualize clusters and their centers from a high-dimensional PyTorch tensor.
    
    Parameters:
        data (torch.Tensor): 2D PyTorch tensor with shape (n_samples, n_features).
        labels (list or torch.Tensor): List or tensor indicating the cluster label for each row in `data`.
        save_path (str): File path to save the plot.
    """
    # Convert data and labels to numpy for easier handling
    data_np = data.numpy()
    labels_np = np.array(labels)
    
    # Perform PCA to reduce data to 2 dimensions
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data_np)
    
    # Calculate cluster centers in the reduced 2D space
    unique_labels = np.unique(labels_np)
    cluster_centers = np.array([data_2d[labels_np == label].mean(axis=0) for label in unique_labels])

    # Plot settings
    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid")

    # Scatter plot for each cluster
    for label in unique_labels:
        cluster_data = data_2d[labels_np == label]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {label}', alpha=0.6)
    
    # Plot cluster centers
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
                color='black', marker='X', s=100, label='Cluster Centers')
    
    # Add titles and labels
    plt.title('Cluster Visualization with Cluster Centers')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()

    # Save the plot to the specified path
    plt.savefig(save_path, format='png')
    plt.close()


device = "cuda:0"

df = pd.read_csv("/workspace/ImputationOT/imputationot/data/CHARTEVENTS_filtered_ICD9.csv") # (3416, 473)
X = df.drop(columns=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE']).iloc[1:]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_scaled)
X = X_reduced

### Imputation
ground_truth = X.copy()
X = torch.tensor(X).to(device)
mask = torch.zeros(X.shape, dtype=torch.bool).to(device)
mask[:1000, :200] = True

nonzero_mask11 = (X[:1000, :200] != 0).cpu()
nonzero_mask21 = (X[1000:, :200] != 0).to(device)
mean_values = torch.sum(X[1000:, :200], dim=0) / torch.sum(nonzero_mask21, dim=0)
imps = mean_values.repeat(1000).to(device)
imps += torch.randn(imps.shape, device=device) * 0.1
imps.requires_grad = True

optimizer = optim.Adam([imps], 0.005)
h_loss = torch.zeros(1).to(device)
lr = 0.01

print("Start optimizing")
for epoch in range(700):
    optimizer.zero_grad()
    X_imputed = X.detach().clone()
    X_imputed[mask] = imps

    X1 = X_imputed[:1000]
    X2 = X_imputed[1000:]
    patients_loss = SamplesLoss()(X1, X2)
    loss = patients_loss
    print(f"{epoch}: patients_loss = {patients_loss.item():.4f}, h_loss = {h_loss.item():.4f}")

    loss.backward()
    optimizer.step()

### Evaluation
X = X_imputed.detach().cpu().numpy()
pcc = pearsonr(X[:1000, :200][nonzero_mask11], ground_truth[:1000, :200][nonzero_mask11])[0]
kmeans = KMeans(n_clusters=len(df['ICD9_CODE'].unique()), random_state=2024)
predicted_clusters = kmeans.fit_predict(X)
true_labels = pd.factorize(df['ICD9_CODE'])[0][1:] 
ari = adjusted_rand_score(true_labels, predicted_clusters)
nmi = normalized_mutual_info_score(true_labels, predicted_clusters)
print(f"Results:")
print(f"  Pearson correlation (PCC): {pcc:.4f}")
print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
print(f"  Normalized Mutual Information (NMI): {nmi:.4f}")
