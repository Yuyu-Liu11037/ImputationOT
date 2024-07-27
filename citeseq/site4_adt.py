import numpy as np
import torch
import torch.optim as optim
import ot
import anndata as ad
import scanpy as sc
import wandb
from scipy.stats import pearsonr

from ..utils import tools

epochs = 100000
device = 'cuda:0'
n_projections = 2000
clustering_interval = 50
K = 9
SITE1_CELL = 16311
SITE2_CELL = 25171
SITE3_CELL = 32029
SITE4_CELL = 16750
FILLED_GEX = 2000

# wandb.init(
#     project="ot",
#
#     config={
#         "dataset": "NIPS2021-Cite-seq",
#         "epochs": epochs,
#         "missing data": "site4 adt",
#         "n_projections": 2000
#     }
# )

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
citeseq = ad.concat([adata_GEX, adata_ADT], axis=1, merge="same")  # X(:,1): GEX, X(:,2): ADT
print(f"Finish preprocessing\n")
#####################################################################################################################################

X = citeseq.X.toarray()
X3 = X[SITE1_CELL + SITE2_CELL:SITE1_CELL + SITE2_CELL + SITE3_CELL].copy()
X = torch.tensor(X).to(device)
X = torch.cat((X[:SITE1_CELL + SITE2_CELL], X[-SITE4_CELL:]),
              dim=0)  # Matrix is too large. Remove certain rows to save memory.
ground_truth = X.clone()

mask = torch.zeros(X.shape, dtype=torch.bool).to(device)
mask[-SITE4_CELL:, 2000:] = True  # mask X(4,2)

nonzero_mask1222 = (X[:SITE1_CELL + SITE2_CELL, 2000:] != 0).to(device)  # nonzero data of X(1,2), X(2,2)
nonzero_mask41 = (X[-SITE4_CELL:, :2000] != 0).to(device)  # nonzero data of X(4,1)
nonzero_mask42 = (X[-SITE4_CELL:, 2000:] != 0).to(device)  # nonzero data of X(4,2)
mean_values = torch.sum(X[:SITE1_CELL + SITE2_CELL, 2000:], dim=0) / torch.sum(nonzero_mask1222, dim=0)
imps = mean_values.repeat(SITE4_CELL).to(device)
imps += torch.randn(imps.shape, device=device) * 0.1
imps.requires_grad = True

optimizer = optim.Adam([imps], lr=0.1)
lambda_lr = lambda epoch: 1 if epoch < 1000 else 0.001 + (0.1 - 0.001) * (1 - (epoch - 1000) / (epochs - 1000))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)


def compute_centers(adata, labels):
    centers = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cell_indices = np.where(labels == label)[0]
        center = adata[cell_indices].X.mean(axis=0)
        centers.append(center)
    return np.array(centers), unique_labels


def compute_masses(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    return counts / counts.sum()


C = torch.randn((2134, K), device=device, requires_grad=True)
print("Start optimizing")
for epoch in range(epochs):
    X_imputed = X.detach().clone()
    X_imputed[mask] = imps

    if epoch == 0:
        pearson_corr = pearsonr(X_imputed[-SITE4_CELL:, 2000:][nonzero_mask42].detach().cpu().numpy(),
                                ground_truth[-SITE4_CELL:, 2000:][nonzero_mask42].detach().cpu().numpy())[0]
        citeseq.X = np.vstack((X_imputed[:SITE1_CELL + SITE2_CELL].detach().cpu().numpy(), X3,
                               X_imputed[-SITE4_CELL:].detach().cpu().numpy()))
        ari, nmi = tools.clustering(citeseq)
        print(f"Initial pearson: {pearson_corr:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}")
        # wandb.log({"Iteration": epoch + 1, "loss": 0, "pearson": pearson_corr, "ari": ari, "nmi": nmi})

    X12 = X_imputed[:SITE1_CELL + SITE2_CELL, :]
    X4 = X_imputed[-SITE4_CELL:, :]
    GEX = torch.transpose(X_imputed[:, :2000], 0, 1)
    ADT = torch.transpose(X_imputed[:, 2000:], 0, 1)
    # Q12 = sinkhorn(-X12 * C)
    # Q4 = sinkhorn(-X4 * C)
    Q12 = ot.sinkhorn(torch.ones(SITE1_CELL + SITE2_CELL) / (SITE1_CELL + SITE2_CELL), torch.ones(K) / K, -torch.mm(X12, C), 1e3)
    Q4 = ot.sinkhorn(torch.ones(SITE4_CELL) / SITE4_CELL, torch.ones(K) / K, -torch.mm(X4, C), 1e3)
    loss = (0.4 * ot.sliced_wasserstein_distance(X12, X4, n_projections=n_projections) +
            0.4 * ot.sliced_wasserstein_distance(GEX, ADT, n_projections=n_projections) +
            0.2 * ot.sliced_wasserstein_distance(Q12, Q4))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    X_imputed = X.detach().clone()
    X_imputed[mask] = imps

    if (epoch + 1) % 300 == 0:
        pearson_corr = pearsonr(X_imputed[-SITE4_CELL:, 2000:][nonzero_mask42].detach().cpu().numpy(),
                                ground_truth[-SITE4_CELL:, 2000:][nonzero_mask42].detach().cpu().numpy())[0]
        citeseq.X = np.vstack((X_imputed[:SITE1_CELL + SITE2_CELL].detach().cpu().numpy(), X3,
                               X_imputed[-SITE4_CELL:].detach().cpu().numpy()))
        ari, nmi = tools.clustering(citeseq)
        print(
            f"Iteration {epoch + 1}/{epochs}: loss: {loss.item():.4f}, pearson: {pearson_corr:.4f}, ari: {ari:.4f}, "
            f"nmi: {nmi:.4f}")
        # wandb.log({"Iteration": epoch + 1, "loss": loss, "pearson": pearson_corr, "ari": ari, "nmi": nmi})
