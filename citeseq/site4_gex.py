import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ot
import sys
import anndata
import scanpy as sc
import wandb
from scipy.stats import pearsonr

wandb.init(
    project="ot",

    config={
    "dataset": "NIPS2021-Cite-seq",
    "epochs": 1000000,
    }
)

epochs = 1000000
device = 'cuda:0'

citeseq = anndata.read_h5ad("./../data/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad")
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
X = torch.cat((X[:41482], X[-16750:]), dim=0)   # Matrix is too large. Remove certain rows to save memory.
ground_truth = X.clone()

mask = torch.zeros(X.shape, dtype=torch.bool).to(device)
# TODO: 前一种编码的问题？
# mask[site3_indices, :][:, adt_indices] = False   # mask X(3,2)
mask[-16750:, :2000] = True   # mask X(4,1)

nonzero_mask1121 = (X[:41482, :2000] != 0).to(device)   # nonzero data of X(1,1), X(2,1)
nonzero_mask41 = (X[-16750:, :][:, :2000] != 0).to(device)   # nonzero data of X(4,1)
nonzero_mask42 = (X[-16750:, :][:, -134:] != 0).to(device)   # nonzero data of X(4,2)
mean_values = torch.sum(X[:41482, :2000], dim=0) / torch.sum(nonzero_mask1121, dim=0)
imps = mean_values.repeat(16750).to(device)
imps += torch.randn(imps.shape, device=device) * 0.1
imps.requires_grad = True

optimizer = optim.Adam([imps], lr=0.1)
lambda_lr = lambda epoch: 1 if epoch < 1000 else 0.001 + (0.1 - 0.001) * (1 - (epoch - 1000) / (epochs - 1000))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

print("start optimizing")
with open('results_bio.txt', 'w') as f:
    for epoch in range(epochs):
        X_imputed = X.detach().clone()
        X_imputed[mask] = imps

        if epoch == 0:
            pearson_corr = pearsonr(X_imputed[-16750:, :2000][nonzero_mask41].detach().cpu().numpy(), ground_truth[-16750:, :2000][nonzero_mask41].detach().cpu().numpy())[0]
            f.write(f"pearson: {pearson_corr:.4f}\n")
            f.flush()

        X12 = X_imputed[:41482, :]
        X4  = X_imputed[-16750:, :]
        GEX = torch.transpose(X_imputed[:, :2000], 0, 1)
        ADT = torch.transpose(X_imputed[:, 2000:], 0, 1)
        loss = 0.5 * ot.sliced_wasserstein_distance(X12, X4) + 0.5 * ot.sliced_wasserstein_distance(GEX, ADT)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        X_imputed = X.detach().clone()
        X_imputed[mask] = imps

        if (epoch + 1) % 200 == 0:
            pearson_corr = pearsonr(X_imputed[-16750:, :2000][nonzero_mask41].detach().cpu().numpy(), ground_truth[-16750:, :2000][nonzero_mask41].detach().cpu().numpy())[0]
            wandb.log({"Iteration": epoch + 1, "loss": loss, "pearson": pearson_corr})
            f.write(f"Iteration {epoch + 1}/{epochs}: loss: {loss.item():.4f}, pearson: {pearson_corr:.4f}\n")
            f.flush()
            