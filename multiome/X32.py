import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ot
import sys
import anndata
import scanpy as sc
import wandb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import pearsonr

# wandb.init(
#     project="ot",

#     config={
#     "dataset": "NIPS2021-Multiome",
#     "epochs": 30000,
#     }
# )

epochs = 30000
device = 'cuda:0'

multiome = anndata.read_h5ad("/workspace/ImputationOT/data/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad")
multiome.var_names_make_unique()

### preprocess
#####################################################################################################################################
### step 1: normalize
adata_GEX = multiome[:, multiome.var['feature_types'] == 'GEX'].copy()
adata_ATAC = multiome[:, multiome.var['feature_types'] == 'ATAC'].copy()
sc.pp.normalize_total(adata_GEX, target_sum=1e4)
sc.pp.normalize_total(adata_ATAC, target_sum=1e4)
citeseq.X[:, multiome.var['feature_types'] == 'GEX'] = adata_GEX.X
citeseq.X[:, multiome.var['feature_types'] == 'ATAC'] = adata_ATAC.X
### step 2: log transform
sc.pp.log1p(multiome)
### step 3
adata_GEX = multiome[:, multiome.var['feature_types'] == 'GEX'].copy()
sc.pp.highly_variable_genes(
    adata_GEX,
    n_top_genes=2000,
    subset=True
)
adata_ATAC = multiome[:, multiome.var['feature_types'] == 'ATAC'].copy()
sc.pp.highly_variable_genes(
    adata_ATAC,
    n_top_genes=8000,
    subset=True
)
### TODO: debug, adata_ATAC contains 8006 peaks
adata_ATAC = adata_ATAC[:, :8000]
highly_variable_peaks_mask = adata_ATAC.var['highly_variable']
highly_variable_genes_mask = adata_GEX.var['highly_variable']
multiome = multiome[:, highly_variable_peaks_mask | highly_variable_genes_mask]
print(multiome)
sys.exit()
#####################################################################################################################################

X = multiome.X.toarray()
X = torch.tensor(X).to(device)
X = X[:47025]   # Matrix is too large. Remove certain rows to save memory.
### TODO: ground_truth values are repeated
ground_truth = X.clone()

mask = torch.zeros(X.shape, dtype=torch.bool).to(device)
mask[-14556:, 8000:] = True   # mask X(3,2)

nonzero_mask1222 = (X[:32469, 8000:] != 0).to(device)   # nonzero data of X(1,2), X(2,2)
nonzero_mask31 = (X[-14556:, :8000] != 0).to(device)   # nonzero data of X(3,1)
nonzero_mask32 = (X[-14556:, 8000:] != 0).to(device)   # nonzero data of X(3,2)
print(ground_truth[-14556:, 8000:][nonzero_mask32])
sys.exit()
mean_values = torch.sum(X[:32469, 8000:], dim=0) / torch.sum(nonzero_mask1222, dim=0)
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
        pearson_corr = pearsonr(X_imputed[-14556:, 8000:][nonzero_mask32].detach().cpu().numpy(), ground_truth[-14556:, 8000:][nonzero_mask32].detach().cpu().numpy())[0]
        print(f"pearson: {pearson_corr:.4f}\n")

    X12 = X_imputed[:32469, :]
    X3  = X_imputed[-14556:, :]
    ATAC = torch.transpose(X_imputed[:, :8000], 0, 1)
    GEX  = torch.transpose(X_imputed[:, 8000:], 0, 1)
    loss = 0.5 * ot.sliced_wasserstein_distance(X12, X3) + 0.5 * ot.sliced_wasserstein_distance(GEX, ATAC)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    X_imputed = X.detach().clone()
    X_imputed[mask] = imps

    if (epoch + 1) % 200 == 0:
        pearson_corr = pearsonr(X_imputed[-14556:, 8000:][nonzero_mask32].detach().cpu().numpy(), ground_truth[-14556:, 8000:][nonzero_mask32].detach().cpu().numpy())[0]
        # wandb.log({"Iteration": epoch + 1, "loss": loss, "pearson": pearson_corr})
        print(f"Iteration {epoch + 1}/{epochs}: loss: {loss.item():.4f}, pearson: {pearson_corr:.4f}\n")
