import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ot
import sys
import anndata
import scanpy as sc
from scipy.stats import pearsonr

epochs = 100000
device = 'cuda:0'
### batch_size <= min(X3[0], X4[0])
batch_size = 6000

citeseq = anndata.read_h5ad("./data/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad")
citeseq.var_names_make_unique()

### preprocess
### step 1
adata_GEX = citeseq[:, citeseq.var['feature_types'] == 'GEX'].copy()
adata_ADT = citeseq[:, citeseq.var['feature_types'] == 'ADT'].copy()
sc.pp.normalize_total(adata_GEX, target_sum=1e4)
sc.pp.normalize_total(adata_ADT, target_sum=1e4)
citeseq.X[:, citeseq.var['feature_types'] == 'GEX'] = adata_GEX.X
citeseq.X[:, citeseq.var['feature_types'] == 'ADT'] = adata_ADT.X
### step 2
sc.pp.log1p(citeseq)
### step 3
adata_GEX = citeseq[:, citeseq.var['feature_types'] == 'GEX'].copy()
sc.pp.highly_variable_genes(
    adata_GEX,
    n_top_genes=2000,
    subset=True
)
highly_variable_genes_mask = adata_GEX.var['highly_variable']
citeseq = citeseq[:, (citeseq.var['feature_types'] == 'ADT') | highly_variable_genes_mask]

X = citeseq.X.toarray()
X = torch.tensor(X).to(device)
X = X[:73511]   # Matrix is too large. Remove certain rows to save memory.
ground_truth = X.clone()

mask = torch.zeros(X.shape, dtype=torch.bool).to(device)
# TODO: 前一种编码的问题？
# mask[site3_indices, :][:, adt_indices] = True   # mask X(3,2)
mask[41482:73511, 13953:] = True   # mask X(3,2)

nonzero_mask1222 = (X[:41482, 13953:] != 0).to(device)   # nonzero data of X(1,2), X(2,2)
nonzero_mask31 = (X[-32029:, :13953] != 0).to(device)   # nonzero data of X(3,1)
nonzero_mask32 = (X[-32029:, 13953:] != 0).to(device)   # nonzero data of X(3,2)
mean_values = torch.sum(X[:41482, 13953:], dim=0) / torch.sum(nonzero_mask1222, dim=0)
imps = mean_values.repeat(32029).to(device)
imps += torch.randn(imps.shape, device=device) * 0.1
imps.requires_grad = True

optimizer = optim.Adam([imps], lr=0.1)

print("start optimizing")
with open('results_bio.txt', 'w') as f:
    for epoch in range(epochs):
        X_imputed = X.detach().clone()
        X_imputed[mask] = imps

        if epoch == 0:
            pearson_corr = pearsonr(X_imputed[-32029:, 13953:][nonzero_mask32].detach().cpu().numpy(), ground_truth[-32029:, 13953:][nonzero_mask32].detach().cpu().numpy())[0]
            f.write(f"pearson: {pearson_corr:.4f}\n")
            f.flush()

        indices1 = torch.randperm(len(site1_indices) + len(site2_indices), device=device)[:batch_size]
        X12 = X_imputed[:(len(site1_indices) + len(site2_indices)), :][indices1, :]
        indices3 = torch.randperm(32029, device=device)[:batch_size]
        X3 = X_imputed[-32029:, :][indices3, :]

        # indices2 = torch.randperm(len(gex_indices), device=device)[:134]
        # GEX = X_imputed[:, indices2]
        # ADT = X_imputed[:, -134:]
        # loss = 0.5 * ot.sliced_wasserstein_distance(X12, X3) + 0.5 * ot.sliced_wasserstein_distance(GEX, ADT)
        loss = ot.sliced_wasserstein_distance(X12, X3)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        X_imputed = X.detach().clone()
        X_imputed[mask] = imps

        if (epoch + 1) % 200 == 0:
            pearson_corr = pearsonr(X_imputed[-32029:, 13953:][nonzero_mask32].detach().cpu().numpy(), ground_truth[-32029:, 13953:][nonzero_mask32].detach().cpu().numpy())[0]
            f.write(f"Iteration {epoch + 1}/{epochs}: loss: {loss.item():.4f}, pearson: {pearson_corr:.4f}\n")
            f.flush()