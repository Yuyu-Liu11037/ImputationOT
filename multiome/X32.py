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
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import pearsonr
from tqdm import tqdm

# wandb.init(
#     project="ot",

#     config={
#     "dataset": "NIPS2021-Multiome",
#     "epochs": 30000,
#     }
# )

epochs = 30000
device = 'cuda:0'

multiome = ad.read_h5ad("/workspace/ImputationOT/data/multiome_processed.h5ad")
multiome.var_names_make_unique()

#####################################################################################################################################
### preprocess
### step 1: normalize
adata_GEX = multiome[:, multiome.var['feature_types'] == 'GEX'].copy()
adata_ATAC = multiome[:, multiome.var['feature_types'] == 'ATAC'].copy()
sc.pp.normalize_total(adata_GEX, target_sum=1e4)
sc.pp.normalize_total(adata_ATAC, target_sum=1e4)
### step 2: log transform
sc.pp.log1p(adata_GEX)
sc.pp.log1p(adata_ATAC)
### step 3
sc.pp.highly_variable_genes(
    adata_GEX,
    subset=True
)
sc.pp.highly_variable_genes(
    adata_ATAC,
    n_top_genes=4000,
    subset=True
)
print('Finished preprocessing')
#####################################################################################################################################

multiome = ad.concat([adata_ATAC, adata_GEX], axis=1)   # changed the original data: left 4000: ATAC, right 2832: GEX
X = multiome.X.toarray()
X = torch.tensor(X).to(device)
X = X[:47025]   # Matrix is too large. Remove certain rows to save memory.
ground_truth = X.clone()

mask = torch.zeros(X.shape, dtype=torch.bool).to(device)
mask[-14556:, 4000:] = True   # mask X(3,2)
nonzero_mask1222 = (X[:32469, 4000:] != 0).to(device)   # nonzero data of X(1,2), X(2,2)
nonzero_mask31 = (X[-14556:, :4000] != 0).to(device)   # nonzero data of X(3,1)
nonzero_mask32 = (X[-14556:, 4000:] != 0).to(device)   # nonzero data of X(3,2)

mean_values = torch.sum(X[:32469, 4000:], dim=0) / torch.sum(nonzero_mask1222, dim=0)
imps = mean_values.repeat(14556).to(device)
imps += torch.randn(imps.shape, device=device) * 0.1
imps.requires_grad = True

optimizer = optim.Adam([imps], lr=0.1)
lambda_lr = lambda epoch: 1 if epoch < 1000 else 0.001 + (0.1 - 0.001) * (1 - (epoch - 1000) / (epochs - 1000))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
mse = sklearn.metrics.mean_squared_error

print("Start optimizing")
for epoch in range(epochs):
    X_imputed = X.detach().clone()
    X_imputed[mask] = imps

    if epoch == 0:
        pearson_corr = pearsonr(X_imputed[-14556:, 4000:][nonzero_mask32].detach().cpu().numpy(), ground_truth[-14556:, 4000:][nonzero_mask32].detach().cpu().numpy())[0]
        rmse_val = np.sqrt(mse(X_imputed[-14556:, 4000:][nonzero_mask32].detach().cpu().numpy(), ground_truth[-14556:, 4000:][nonzero_mask32].detach().cpu().numpy()))
        print(f"pearson: {pearson_corr:.4f}, rmse: {rmse_val:.4f}")

    X12 = X_imputed[:32469, :]
    X3  = X_imputed[-14556:, :]
    ATAC = torch.transpose(X_imputed[:, :4000], 0, 1)
    GEX  = torch.transpose(X_imputed[:, 4000:], 0, 1)
    loss = 0.5 * ot.sliced_wasserstein_distance(X12, X3) + 0.5 * ot.sliced_wasserstein_distance(GEX, ATAC)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 200 == 0:
        X_imputed = X.detach().clone()
        X_imputed[mask] = imps
        
        pearson_corr = pearsonr(X_imputed[-14556:, 4000:][nonzero_mask32].detach().cpu().numpy(), ground_truth[-14556:, 4000:][nonzero_mask32].detach().cpu().numpy())[0]
        rmse_val = np.sqrt(mse(X_imputed[-14556:, 4000:][nonzero_mask32].detach().cpu().numpy(), ground_truth[-14556:, 4000:][nonzero_mask32].detach().cpu().numpy()))
        # wandb.log({"Iteration": epoch + 1, "loss": loss, "pearson": pearson_corr})
        print(f"Iteration {epoch + 1}/{epochs}: loss: {loss.item():.4f}, pearson: {pearson_corr:.4f}, rmse: {rmse_val:.4f}")
