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
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import pearsonr

# wandb.init(
#     project="ot",

#     config={
#     "dataset": "NIPS2021-Cite-seq",
#     "epochs": 30000,
#     }
# )

epochs = 30000
device = 'cuda:0'

citeseq = anndata.read_h5ad("/workspace/ImputationOT/data/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad")
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

def clustering(
    adata,
    resolutions,
    clustering_method,
    cell_type_col,
    batch_col,
):
    """Clusters the data and calculate agreement with cell type and batch
    variable.

    This method cluster the neighborhood graph (requires having run sc.pp.
    neighbors first) with "clustering_method" algorithm multiple times with the
    given resolutions, and return the best result in terms of ARI with cell
    type.
    Other metrics such as NMI with cell type, ARi with batch are logged but not
    returned. (TODO: also return these metrics)

    Args:
        adata: the dataset to be clustered. adata.obsp shouhld contain the keys
            'connectivities' and 'distances'.
        resolutions: a list of leiden/louvain resolution parameters. Will
            cluster with each resolution in the list and return the best result
            (in terms of ARI with cell type).
        clustering_method: Either "leiden" or "louvain".
        cell_type_col: a key in adata.obs to the cell type column.
        batch_col: a key in adata.obs to the batch column.

    Returns:
        best_cluster_key: a key in adata.obs to the best (in terms of ARI with
            cell type) cluster assignment column.
        best_ari: the best ARI with cell type.
        best_nmi: the best NMI with cell type.
    """

    assert len(resolutions) > 0, f"Must specify at least one resolution."

    if clustering_method == "leiden":
        clustering_func = sc.tl.leiden
    elif clustering_method == "louvain":
        clustering_func = sc.tl.louvain
    else:
        raise ValueError(
            "Please specify louvain or leiden for the clustering method argument."
        )
        
    assert cell_type_col in adata.obs, f"{cell_type_col} not in adata.obs"
    best_res, best_ari, best_nmi = None, -inf, -inf
    for res in resolutions:
        col = f"{clustering_method}_{res}"
        clustering_func(adata, resolution=res, key_added=col)
        ari = adjusted_rand_score(adata.obs[cell_type_col], adata.obs[col])
        nmi = normalized_mutual_info_score(adata.obs[cell_type_col], adata.obs[col])
        n_unique = adata.obs[col].nunique()
        if ari > best_ari:
            best_res = res
            best_ari = ari
        if nmi > best_nmi:
            best_nmi = nmi
        if batch_col in adata.obs and adata.obs[batch_col].nunique() > 1:
            ari_batch = adjusted_rand_score(adata.obs[batch_col], adata.obs[col])
            # print(f'Resolution: {res:5.3g}\tARI: {ari:7.4f}\tNMI: {nmi:7.4f}\tbARI: {ari_batch:7.4f}\t# labels: {n_unique}')
        else:
            # print(f'Resolution: {res:5.3g}\tARI: {ari:7.4f}\tNMI: {nmi:7.4f}\t# labels: {n_unique}')
            a = None

    return f"{clustering_method}_{best_res}", best_ari, best_nmi

scanpy.pp.neighbors(citeseq)
print(clustering(citeseq, np.arange(0.75, 2, 0.1)))
sys.exit()

print("start optimizing")
for epoch in range(epochs):
    X_imputed = X.detach().clone()
    X_imputed[mask] = imps
    if epoch == 0:
        pearson_corr = pearsonr(X_imputed[:16311, :2000][nonzero_mask].detach().cpu().numpy(), ground_truth[:16311, :2000][nonzero_mask].detach().cpu().numpy())[0]
        print(f"Initial pearson: {pearson_corr:.4f}\n")

    X1 = X_imputed[:16311, :]
    X2 = X_imputed[16311:, :]
    GEX = torch.transpose(X_imputed[:, :2000], 0, 1)
    ADT = torch.transpose(X_imputed[:, 2000:], 0, 1)
    loss = 0.5 * ot.sliced_wasserstein_distance(X1, X2) + 0.5 * ot.sliced_wasserstein_distance(GEX, ADT)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    X_imputed = X.detach().clone()
    X_imputed[mask] = imps

    if (epoch + 1) % 200 == 0:
        pearson_corr = pearsonr(X_imputed[:16311, :2000][nonzero_mask].detach().cpu().numpy(), ground_truth[:16311, :2000][nonzero_mask].detach().cpu().numpy())[0]
        wandb.log({"Iteration": epoch + 1, "loss": loss, "pearson": pearson_corr})
        print(f"Iteration {epoch + 1}/{epochs}: loss: {loss.item():.4f}, pearson: {pearson_corr:.4f}\n")
