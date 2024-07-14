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
threshold = .5

multiome = ad.read_h5ad("/workspace/ImputationOT/data/multiome_processed.h5ad")
multiome.var_names_make_unique()

#####################################################################################################################################
### preprocess
adata_GEX = multiome[:, multiome.var['feature_types'] == 'GEX'].copy()
adata_ATAC = multiome[:, multiome.var['feature_types'] == 'ATAC'].copy()
### step 1: normalize
# sc.pp.normalize_total(adata_GEX, target_sum=1e6)
# sc.pp.normalize_total(adata_ATAC, target_sum=1e6)
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
print('Finished preprocessing')
print()
X = adata_ATAC.X
print("ATAC data after filtering")
print("Matrix Shape:", X.shape)
print("Density:", X.nnz / (X.shape[0] * X.shape[1]))
print("Minimum Value:", X.min())
print("Maximum Value:", X.max())
print("Mean Value without zeros:", X[X != 0].mean())
print()

X = adata_GEX.X
print("GEX data after filtering")
print("Matrix Shape:", X.shape)
print("Density:", X.nnz / (X.shape[0] * X.shape[1]))
print("Minimum Value:", X.min())
print("Maximum Value:", X.max())
print("Mean Value without zeros:", X[X != 0].mean())
print()

num_atac = adata_ATAC.X.shape[1]
multiome = ad.concat([adata_ATAC, adata_GEX], axis=1)   # changed the original data: left num_atac: ATAC, right 2832: GEX
print(multiome.X.shape)
print()
#####################################################################################################################################

X = multiome.X.toarray()
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
rmse = sklearn.metrics.root_mean_squared_error

print("start optimizing")
for epoch in range(epochs):
    X_imputed = X.detach().clone()
    X_imputed[mask] = imps

    if epoch == 0:
        rmse_val = rmse(X_imputed[-14556:, :num_atac][nonzero_mask31].detach().cpu().numpy(), ground_truth[-14556:, :num_atac][nonzero_mask31].detach().cpu().numpy())
        print(f"Initial pearson: nan, rmse: {rmse_val:.4f}")

    X12 = X_imputed[:47025, :]
    X4  = X_imputed[-14556:, :]
    ATAC = torch.transpose(X_imputed[:, :num_atac], 0, 1)
    GEX = torch.transpose(X_imputed[:, num_atac:], 0, 1)
    loss = 0.5 * ot.sliced_wasserstein_distance(X12, X4) + 0.5 * ot.sliced_wasserstein_distance(GEX, ATAC)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 200 == 0:
        X_imputed = X.detach().clone()
        X_imputed[mask] = imps

        rmse_val = rmse(X_imputed[-14556:, :num_atac][nonzero_mask31].detach().cpu().numpy(), ground_truth[-14556:, :num_atac][nonzero_mask31].detach().cpu().numpy())
        # wandb.log({"Iteration": epoch + 1, "loss": loss, "rmse": rmse_val})
        print(f"Iteration {epoch + 1}/{epochs}: loss: {loss.item():.4f}, pearson: nan, rmse: {rmse_val:.4f}")