import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ot
import sys
# import wandb
import anndata
import scipy.stats as stats
from scipy.stats import pearsonr
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler, SimpleFill


epochs = 1000000
device = 'cuda:0'
### batch_size <= min(X1[0], X2[0])
batch_size = 8000

citeseq = anndata.read_h5ad("/workspace/ImputationOT/citeseq_processed-001.h5ad")
# print(type(citeseq.X))

X = citeseq.X.toarray()
X = np.log1p(X)
X = torch.tensor(X).to(device)
X = X[:41482]    # data in site1, site2
ground_truth = X.clone()

gex_indices = np.where(citeseq.var['feature_types'] == 'GEX')[0]
adt_indices = np.where(citeseq.var['feature_types'] == 'ADT')[0]
GEX = X[:41482, gex_indices] # AnnData object with n_obs × n_vars = 41482 × 13953
ADT = X[:41482, adt_indices] # AnnData object with n_obs × n_vars = 41482 × 134

site1_indices = np.where(citeseq.obs['Site'] == 'site1')[0]
site2_indices = np.where(citeseq.obs['Site'] == 'site2')[0]
X1 = X[site1_indices, :] # AnnData object with n_obs × n_vars = 16311 × 14087
X2 = X[site2_indices, :] # AnnData object with n_obs × n_vars = 25171 × 14087

### 把已有数据也当作0数据进行全面填补
mask = torch.ones((41482, 14087), dtype=torch.bool).to(device)
mask[:16311, :13953] = False   # mask X(1,1)
mask = ~mask

nonzero_mask = (X[:16311, :13953] != 0).to(device)   # nonzero data of X(1,1)
nonzero_mask2 = (X[:16311, 13953:] != 0).to(device)   # nonzero data of X(1,2)

mean_values = torch.sum(X[:16311, 13953:] * nonzero_mask2, dim=1) / torch.sum(nonzero_mask, dim=1)
imps = mean_values.repeat(13953).to(device)   # shape = [16311, 13953]
imps.requires_grad = True
# imps = torch.ones(mask.sum(), device=device, requires_grad=True)
optimizer = optim.Adam([imps])

print("start optimizing")
with open('results_bio.txt', 'w') as f:
    for epoch in range(epochs):
        X_imputed = X.detach().clone()
        X_imputed[mask] = imps

        indices1 = torch.randperm(16311 // 2, device=device)[:batch_size]
        X1 = X_imputed[:16311, :][indices1, :]
        X2 = X_imputed[16311:, :][indices1, :]

        indices2 = torch.randperm(13953, device=device)[:134]
        GEX = X_imputed[:, indices2]
        ADT = X_imputed[:, -134:]
        loss = 0.5 * ot.sliced_wasserstein_distance(X1, X2) + 0.5 * ot.sliced_wasserstein_distance(GEX, ADT)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        X_imputed = X.detach().clone()
        X_imputed[mask] = imps

        if (epoch + 1) % 100 == 0:
            pearson_corr = pearsonr(X_imputed[:16311, :13953][nonzero_mask].detach().cpu().numpy(), ground_truth[:16311, :13953][nonzero_mask].detach().cpu().numpy())[0]
            f.write(f"Iteration {epoch + 1}/{epochs}: loss: {loss.item():.4f}, pearson: {pearson_corr:4f}\n")
            f.flush()