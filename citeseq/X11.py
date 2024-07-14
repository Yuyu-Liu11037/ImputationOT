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
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import pearsonr
from tqdm import tqdm

# wandb.init(
#     project="ot",

#     config={
#     "dataset": "NIPS2021-Cite-seq",
#     "epochs": 30000,
#     }
# )

epochs = 30000
device = "cuda:0"

citeseq = ad.read_h5ad("/workspace/ImputationOT/data/citeseq_processed.h5ad")
citeseq.var_names_make_unique()

#####################################################################################################################################
### preprocess
adata_GEX = citeseq[:, citeseq.var["feature_types"] == "GEX"].copy()
adata_ADT = citeseq[:, citeseq.var["feature_types"] == "ADT"].copy()
### step 1
sc.pp.normalize_total(adata_GEX, target_sum=1e4)
sc.pp.normalize_total(adata_ADT, target_sum=1e4)
### step 2
sc.pp.log1p(adata_GEX)
sc.pp.log1p(adata_ADT)
### step 3
sc.pp.highly_variable_genes(
    adata_GEX,
    n_top_genes=2000,
    subset=True
)
citeseq = ad.concat([adata_GEX, adata_ADT], axis=1, merge="first") 
print(f"Finished preprocessing\n")
#####################################################################################################################################

def clustering(adata):
    sc.pp.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
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

print("Ground truth clustering")
ari, nmi = clustering(citeseq)
print(f"ari: {ari:.4f}, nmi: {nmi:.4f}\n")

X = citeseq.X.toarray()
X34 = X[41482:]
X = torch.tensor(X).to(device)
X = X[:41482]    # data in site1, site2
ground_truth = X.clone()

mask = torch.zeros((41482, 2134), dtype=torch.bool).to(device)
mask[:16311, :2000] = True   # mask X(1,1)

nonzero_mask  = (X[:16311, :2000] != 0).to(device)   # nonzero data of X(1,1)
nonzero_mask2 = (X[:16311, 2000:] != 0).to(device)   # nonzero data of X(1,2)
nonzero_mask3 = (X[16311:, :2000] != 0).to(device)   # nonzero data of X(2,1)

mean_values = torch.sum(X[16311:, :2000], dim=0) / torch.sum(nonzero_mask3, dim=0)
imps = mean_values.repeat(16311).to(device)   # shape = [16311, 2000]
imps += torch.randn(imps.shape, device=device) * 0.1
imps.requires_grad = True

optimizer = optim.Adam([imps], lr=0.1)
lambda_lr = lambda epoch: 1 if epoch < 1000 else 0.001 + (0.1 - 0.001) * (1 - (epoch - 1000) / (epochs - 1000))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
rmse = sklearn.metrics.root_mean_squared_error

print("Start optimizing")
for epoch in range(epochs):
    X_imputed = X.detach().clone()
    X_imputed[mask] = imps
    if epoch == 0:
        rmse_val = rmse(X_imputed[:16311, :2000][nonzero_mask].detach().cpu().numpy(), ground_truth[:16311, :2000][nonzero_mask].detach().cpu().numpy())
        pearson_corr = pearsonr(X_imputed[:16311, :2000][nonzero_mask].detach().cpu().numpy(), ground_truth[:16311, :2000][nonzero_mask].detach().cpu().numpy())[0]
        print(f"Initial rmse: {rmse_val:.4f}, pearson: {pearson_corr:.4f}")
        
    X1 = X_imputed[:16311, :]
    X2 = X_imputed[16311:, :]
    GEX = torch.transpose(X_imputed[:, :2000], 0, 1)
    ADT = torch.transpose(X_imputed[:, 2000:], 0, 1)
    loss = 0.5 * ot.sliced_wasserstein_distance(X1, X2) + 0.5 * ot.sliced_wasserstein_distance(GEX, ADT)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 250 == 0:
        X_imputed = X.detach().clone()
        X_imputed[mask] = imps

        rmse_val = rmse(X_imputed[:16311, :2000][nonzero_mask].detach().cpu().numpy(), ground_truth[:16311, :2000][nonzero_mask].detach().cpu().numpy())
        pearson_corr = pearsonr(X_imputed[:16311, :2000][nonzero_mask].detach().cpu().numpy(), ground_truth[:16311, :2000][nonzero_mask].detach().cpu().numpy())[0]
        citeseq.X = np.vstack((X_imputed.detach().cpu().numpy(), X34))
        ari, nmi = clustering(citeseq)
        print(f"Iteration {epoch + 1}/{epochs}: loss: {loss.item():.4f}, rmse:{rmse_val:.4f}, pearson: {pearson_corr:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}")
        # wandb.log({"Iteration": epoch + 1, "loss": loss, "rmse": rmse_val, "pearson": pearson_corr, "ari": ari, "nmi": nmi})