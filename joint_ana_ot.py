import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ot
import sys
import anndata
import scipy.stats as stats
from scipy.stats import pearsonr
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler, SimpleFill


epochs = 1000000
device = 'cuda:0'
### batch_size <= min(X3[0], X4[0])
batch_size = 16000

citeseq = anndata.read_h5ad("./data/citeseq_processed-001.h5ad")

X = citeseq.X.toarray()
X = np.log1p(X)
X = torch.tensor(X).to(device)
ground_truth = X.clone()

gex_indices = np.where(citeseq.var['feature_types'] == 'GEX')[0]
adt_indices = np.where(citeseq.var['feature_types'] == 'ADT')[0]
GEX = X[:41482, gex_indices] # AnnData object with n_obs × n_vars = 41482 × 13953
ADT = X[:41482, adt_indices] # AnnData object with n_obs × n_vars = 41482 × 134
site1_indices = np.where(citeseq.obs['Site'] == 'site1')[0]
site2_indices = np.where(citeseq.obs['Site'] == 'site2')[0]
site3_indices = np.where(citeseq.obs['Site'] == 'site3')[0]
site4_indices = np.where(citeseq.obs['Site'] == 'site4')[0]
X1 = X[site1_indices, :] # AnnData object with n_obs × n_vars = 16311 × 14087
X3 = X[site2_indices, :] # AnnData object with n_obs × n_vars = 25171 × 14087
X3 = X[site3_indices, :]
X4 = X[site4_indices, :]

mask = torch.ones(X.shape, dtype=torch.bool).to(device)
# TODO: 前一种编码的问题？
# mask[site3_indices, :][:, adt_indices] = False   # mask X(3,2)
# mask[41482:73511, 13953:] = False   # mask X(3,2)
mask[41482:73511, :13953] = False   # mask X(3,1)
mask = ~mask

nonzero_mask31 = (X[site3_indices, :][:, gex_indices] != 0).to(device)   # nonzero data of X(3,1)
nonzero_mask32 = (X[site3_indices, :][:, adt_indices] != 0).to(device)   # nonzero data of X(3,2)
nonzero_mask41 = (X[site4_indices, :][:, gex_indices] != 0).to(device)   # nonzero data of X(4,1)
nonzero_mask42 = (X[site4_indices, :][:, adt_indices] != 0).to(device)   # nonzero data of X(4,2)
mean_values3 = torch.sum(X[41482:73511, 13953:] * nonzero_mask32, dim=1) / torch.sum(nonzero_mask32, dim=1)
imps = mean_values3.repeat(len(adt_indices)).to(device)
imps.requires_grad = True

optimizer = optim.Adam([imps])

print("start optimizing")
with open('results_bio.txt', 'w') as f:
    for epoch in range(epochs):
        X_imputed = X.detach().clone()
        X_imputed[mask] = imps

        indices1 = torch.randperm((len(site1_indices) + len(site2_indices)) // 2, device=device)[:batch_size]
        X12 = X_imputed[:(len(site1_indices) + len(site2_indices)), :][indices1, :]
        X3 = X_imputed[site3_indices, :][indices1, :]

        indices2 = torch.randperm(len(gex_indices), device=device)[:134]
        GEX = X_imputed[:, indices2]
        ADT = X_imputed[:, -134:]
        loss = 0.5 * ot.sliced_wasserstein_distance(X12, X3) + 0.5 * ot.sliced_wasserstein_distance(GEX, ADT)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        X_imputed = X.detach().clone()
        X_imputed[mask] = imps

        if (epoch + 1) % 100 == 0:
            pearson_corr = pearsonr(X_imputed[41482:73511, :13953][nonzero_mask32].detach().cpu().numpy(), ground_truth[41482:73511, :13953][nonzero_mask32].detach().cpu().numpy())[0]
            f.write(f"Iteration {epoch + 1}/{epochs}: loss: {loss.item():.4f}, pearson: {pearson_corr:.4f}\n")
            f.flush()
            