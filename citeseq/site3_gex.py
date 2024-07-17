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
SITE1_CELL = 16311
SITE2_CELL = 25171
SITE3_CELL = 32029
SITE4_CELL = 16750
FILLED_GEX = 2000

wandb.init(
    project="ot",

    config={
    "dataset": "NIPS2021-Cite-seq",
    "epochs": epochs,
    "missing data": "site3 gex"
    }
)

citeseq = ad.read_h5ad("/workspace/ImputationOT/data/citeseq_processed.h5ad")
citeseq.var_names_make_unique()

#####################################################################################################################################
### preprocess
adata_GEX = citeseq[:, citeseq.var["feature_types"] == "GEX"].copy()
adata_ADT = citeseq[:, citeseq.var["feature_types"] == "ADT"].copy()
### step 1
sc.pp.normalize_total(adata_GEX)
sc.pp.normalize_total(adata_ADT)
### step 2
sc.pp.log1p(adata_GEX)
sc.pp.log1p(adata_ADT)
### step 3
sc.pp.highly_variable_genes(
    adata_GEX,
    n_top_genes=FILLED_GEX,
    subset=True
)
citeseq = ad.concat([adata_GEX, adata_ADT], axis=1, merge="first")   # X(:,1): GEX, X(:,2): ADT
print(f"Finish preprocessing\n")
#####################################################################################################################################

def clustering(adata):
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, use_rep="X_pca")
    resolution_values = [0.1, 0.5, 0.75, 1.0]
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

X = citeseq.X.toarray()
X4 = X[SITE1_CELL + SITE2_CELL + SITE3_CELL:].copy()
X = torch.tensor(X).to(device)
X = X[:SITE1_CELL + SITE2_CELL + SITE3_CELL]   # Matrix is too large. Remove certain rows to save memory.
ground_truth = X.clone()

mask = torch.zeros(X.shape, dtype=torch.bool).to(device)
mask[SITE1_CELL + SITE2_CELL:SITE1_CELL + SITE2_CELL + SITE3_CELL, :FILLED_GEX] = True   # mask X(3,1)

nonzero_mask1121 = (X[:SITE1_CELL + SITE2_CELL, :FILLED_GEX] != 0).to(device)   # nonzero data of X(1,1), X(2,1)
nonzero_mask31 = (X[-SITE3_CELL:, :FILLED_GEX] != 0).to(device)   # nonzero data of X(3,1)
nonzero_mask32 = (X[-SITE3_CELL:, FILLED_GEX:] != 0).to(device)   # nonzero data of X(3,2)
mean_values = torch.sum(X[:SITE1_CELL + SITE2_CELL, :FILLED_GEX], dim=0) / torch.sum(nonzero_mask1121, dim=0)
imps = mean_values.repeat(SITE3_CELL).to(device)
imps += torch.randn(imps.shape, device=device) * 0.1
imps.requires_grad = True

optimizer = optim.Adam([imps], lr=0.1)
lambda_lr = lambda epoch: 1 if epoch < 1000 else 0.001 + (0.1 - 0.001) * (1 - (epoch - 1000) / (epochs - 1000))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

print("Start optimizing")
for epoch in range(epochs):
    X_imputed = X.detach().clone()
    X_imputed[mask] = imps

    if epoch == 0:
        pearson_corr = pearsonr(X_imputed[-SITE3_CELL:, :FILLED_GEX][nonzero_mask31].detach().cpu().numpy(), ground_truth[-SITE3_CELL:, :FILLED_GEX][nonzero_mask31].detach().cpu().numpy())[0]
        citeseq.X = np.vstack((X_imputed.detach().cpu().numpy(), X4))
        ari, nmi = clustering(citeseq)
        print(f"Initial pearson: {pearson_corr:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}")
        wandb.log({"Iteration": epoch, "loss": 0, "pearson": pearson_corr, "ari": ari, "nmi": nmi})

    X12 = X_imputed[:SITE1_CELL + SITE2_CELL, :]
    X3  = X_imputed[-SITE3_CELL:, :]
    GEX = torch.transpose(X_imputed[:, :FILLED_GEX], 0, 1)
    ADT = torch.transpose(X_imputed[:, FILLED_GEX:], 0, 1)
    loss = 0.5 * ot.sliced_wasserstein_distance(X12, X3) + 0.5 * ot.sliced_wasserstein_distance(GEX, ADT)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 300 == 0:
        X_imputed = X.detach().clone()
        X_imputed[mask] = imps
        
        pearson_corr = pearsonr(X_imputed[-SITE3_CELL:, :FILLED_GEX][nonzero_mask31].detach().cpu().numpy(), ground_truth[-SITE3_CELL:, :FILLED_GEX][nonzero_mask31].detach().cpu().numpy())[0]
        citeseq.X = np.vstack((X_imputed.detach().cpu().numpy(), X4))
        ari, nmi = clustering(citeseq)
        print(f"Iteration {epoch + 1}/{epochs}: loss: {loss.item():.4f}, pearson: {pearson_corr:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}")
        wandb.log({"Iteration": epoch + 1, "loss": loss, "pearson": pearson_corr, "ari": ari, "nmi": nmi})
