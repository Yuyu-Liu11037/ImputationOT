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

# wandb.init(
#     project="ot",

#     config={
#     "dataset": "NIPS2021-Multiome",
#     "epochs": 1000000,
#     }
# )

epochs = 1000000
device = 'cuda:0'
threshold = .5

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
X = torch.cat((X[:32469], X[-22224:]), dim=0)   # Matrix is too large. Remove certain rows to save memory.
ground_truth = X.clone()

mask = torch.zeros(X.shape, dtype=torch.bool).to(device)
mask[-22224:, :4000] = True   # mask X(4,1)

nonzero_mask1121 = (X[:32469, :4000] != 0).to(device)   # nonzero data of X(1,1), X(2,1)
nonzero_mask41 = (X[-22224:, :4000] != 0).to(device)   # nonzero data of X(4,1)
nonzero_mask42 = (X[-22224:, 4000:] != 0).to(device)   # nonzero data of X(4,2)
mean_values = torch.sum(X[:32469, :4000], dim=0) / torch.sum(nonzero_mask1121, dim=0)
imps = mean_values.repeat(22224).to(device)
imps += torch.randn(imps.shape, device=device) * 0.1
imps.requires_grad = True

optimizer = optim.Adam([imps], lr=0.1)
lambda_lr = lambda epoch: 1 if epoch < 1000 else 0.001 + (0.1 - 0.001) * (1 - (epoch - 1000) / (epochs - 1000))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
mse = sklearn.metrics.mean_squared_error

print("start optimizing")
for epoch in range(epochs):
    X_imputed = X.detach().clone()
    X_imputed[mask] = imps

    if epoch == 0:
        # imps_int = (imps > .95).float()
        # X_imputed[mask] = imps_int
        rmse_val = np.sqrt(mse(X_imputed[-22224:, :4000][nonzero_mask41].detach().cpu().numpy(), ground_truth[-22224:, :4000][nonzero_mask41].detach().cpu().numpy()))
        print(f"Initial rmse: {rmse_val:.4f}")

    X12 = X_imputed[:32469, :]
    X4  = X_imputed[-22224:, :]
    ATAC = torch.transpose(X_imputed[:, :4000], 0, 1)
    GEX = torch.transpose(X_imputed[:, 4000:], 0, 1)
    loss = 0.5 * ot.sliced_wasserstein_distance(X12, X4) + 0.5 * ot.sliced_wasserstein_distance(GEX, ATAC)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 200 == 0:
        X_imputed = X.detach().clone()
        X_imputed[mask] = imps
        # imps_int = (imps > threshold).float()
        # X_imputed[mask] = imps_int
            
        rmse_val = np.sqrt(mse(X_imputed[-22224:, :4000][nonzero_mask41].detach().cpu().numpy(), ground_truth[-22224:, :4000][nonzero_mask41].detach().cpu().numpy()))
        # wandb.log({"Iteration": epoch + 1, "loss": loss, "pearson": pearson_corr})
        print(f"Iteration {epoch + 1}/{epochs}: loss: {loss.item():.4f}, rmse: {rmse_val:.4f}")