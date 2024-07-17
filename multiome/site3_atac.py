import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ot
import sys
import anndata as ad
import scanpy as sc
import wandb
import sklearn
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import pearsonr

epochs = 100000
device = 'cuda:0'

wandb.init(
    project="ot",

    config={
        "dataset": "NIPS2021-Multiome",
        "missing data": "site3 atac",
        "epochs": epochs,
        "use_normalization": True,
        "target_sum": 1e4
    }
)

multiome = ad.read_h5ad("/workspace/ImputationOT/data/multiome_processed.h5ad")
multiome.var_names_make_unique()

#####################################################################################################################################
print("Start preprocessing")
### preprocess
adata_GEX = multiome[:, multiome.var['feature_types'] == 'GEX'].copy()
adata_ATAC = multiome[:, multiome.var['feature_types'] == 'ATAC'].copy()
### step 1: normalize
print("Use normalization")
sc.pp.normalize_total(adata_GEX, target_sum=1e4)
sc.pp.normalize_total(adata_ATAC, target_sum=1e4)
### step 2: log transform
sc.pp.log1p(adata_GEX)
sc.pp.log1p(adata_ATAC)
### step 3: select highly variable features
sc.pp.highly_variable_genes(adata_GEX, subset=True)
sc.pp.highly_variable_genes(
    adata_ATAC,
    n_top_genes=4000,
    subset=True
)

num_atac = adata_ATAC.X.shape[1]
multiome = ad.concat([adata_ATAC, adata_GEX], axis=1, merge="first")   # left num_atac: ATAC, right 2832: GEX

print(f"Finished preprocessing\n")
#####################################################################################################################################

def clustering(adata):
    sc.pp.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
    resolution_values = [0.1, 0.5, 0.75]
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
    
X = multiome.X.toarray()
X4 = X[-22224:].copy()
X = torch.tensor(X).to(device)
X = X[:47025]   # Matrix is too large. Remove certain rows to save memory.
ground_truth = X.clone()

mask = torch.zeros(X.shape, dtype=torch.bool).to(device)
mask[-14556:, :num_atac] = True   # mask X(3,1)
nonzero_mask1121 = (X[:32469, :num_atac] != 0).to(device)   # nonzero data of X(1,1), X(2,1)
nonzero_mask31 = (X[-14556:, :num_atac] != 0).to(device)   # nonzero data of X(3,1)
nonzero_mask32 = (X[-14556:, num_atac:] != 0).to(device)   # nonzero data of X(3,2)

mean_values = torch.sum(X[:32469, :num_atac], dim=0) / torch.sum(nonzero_mask1121, dim=0)
imps = mean_values.repeat(14556).to(device)
imps += torch.randn(imps.shape, device=device) * 0.1
imps.requires_grad = True

optimizer = optim.Adam([imps], lr=0.1)
lambda_lr = lambda epoch: 1 if epoch < 1000 else 0.001 + (0.1 - 0.001) * (1 - (epoch - 1000) / (epochs - 1000))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

print("start optimizing")
for epoch in range(epochs):
    X_imputed = X.detach().clone()
    X_imputed[mask] = imps

    if epoch == 0:
        pearson_corr = pearsonr(X_imputed[-14556:, :num_atac][nonzero_mask31].detach().cpu().numpy(), ground_truth[-14556:, :num_atac][nonzero_mask31].detach().cpu().numpy())[0]
        multiome.X = np.vstack((X_imputed.detach().cpu().numpy(), X4))
        ari, nmi = clustering(multiome)
        print(f"Initial pearson: {pearson_corr:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}")
        wandb.log({"Iteration": 0, 
                   "loss": 0, 
                   "pearson": pearson_corr, 
                   "ari": ari, 
                   "nmi": nmi}
                 )

    X12 = X_imputed[:47025, :]
    X3  = X_imputed[-14556:, :]
    ATAC = torch.transpose(X_imputed[:, :num_atac], 0, 1)
    GEX = torch.transpose(X_imputed[:, num_atac:], 0, 1)
    loss = 0.5 * ot.sliced_wasserstein_distance(X12, X3) + 0.5 * ot.sliced_wasserstein_distance(GEX, ATAC)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 300 == 0:
        X_imputed = X.detach().clone()
        X_imputed[mask] = imps

        pearson_corr = pearsonr(X_imputed[-14556:, :num_atac][nonzero_mask31].detach().cpu().numpy(), ground_truth[-14556:, :num_atac][nonzero_mask31].detach().cpu().numpy())[0]
        multiome.X = np.vstack((X_imputed.detach().cpu().numpy(), X4))
        ari, nmi = clustering(multiome)
        print(f"Iteration {epoch + 1}/{epochs}: loss: {loss.item():.4f}, pearson: {pearson_corr:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}")
        wandb.log({"Iteration": epoch + 1, 
                   "loss": loss, 
                   "pearson": pearson_corr, 
                   "ari": ari, 
                   "nmi": nmi}
                 )
        