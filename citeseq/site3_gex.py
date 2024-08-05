import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import ot
import anndata as ad
import scanpy as sc
import wandb
import sys
import random
from scipy.stats import pearsonr

from utils import tools

# 设置随机种子
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

epochs = 8000
device = 'cuda:0'
K = 10
n_projections = 2000
batch_size = 5000
SITE1_CELL = 16311
SITE2_CELL = 25171
SITE3_CELL = 32029
SITE4_CELL = 16750
FILLED_GEX = 2000

wandb.init(
    project="ot",
    name="c-3gex-clt2",
    config={
        "dataset": "NIPS2021-Cite-seq",
        "epochs": epochs,
        "missing data": "site3 gex",
        "n_projections": 2000,
        "h_loss weight": 0.1,
        "n_classes": 10
    }
)

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
citeseq = ad.concat([adata_GEX, adata_ADT], axis=1, merge="first")   # X(:,1): GEX, X(:,2): ADT
print(f"Finish preprocessing\n")
#####################################################################################################################################

X = citeseq.X.toarray()
X4 = X[SITE1_CELL + SITE2_CELL + SITE3_CELL:].copy()
X = torch.tensor(X).to(device)
X = X[:SITE1_CELL + SITE2_CELL + SITE3_CELL]   # Matrix is too large. Remove certain rows to save memory.
ground_truth = X.clone()

mask = torch.zeros(X.shape, dtype=torch.bool).to(device)
mask[SITE1_CELL + SITE2_CELL: SITE1_CELL + SITE2_CELL + SITE3_CELL, :FILLED_GEX] = True   # mask X(3,1)

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
dkm = tools.DKM(num_clusters=K, batch_size=100).cuda()

print("Start optimizing")
for epoch in range(epochs):
    X_imputed = X.detach().clone()
    X_imputed[mask] = imps

    if epoch == 0:
        pearson_corr = pearsonr(X_imputed[-SITE3_CELL:, :FILLED_GEX][nonzero_mask31].detach().cpu().numpy(), ground_truth[-SITE3_CELL:, :FILLED_GEX][nonzero_mask31].detach().cpu().numpy())[0]
        citeseq.X = np.vstack((X_imputed.detach().cpu().numpy(), X4))
        ari, nmi = tools.clustering(citeseq)
        print(f"Initial pearson: {pearson_corr:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}")
        wandb.log({"Iteration": epoch, "loss": 0, "pearson": pearson_corr, "ari": ari, "nmi": nmi})

    indices1 = torch.randperm(SITE1_CELL + SITE2_CELL, device=device)[:batch_size]
    indices2 = torch.randperm(SITE3_CELL, device=device)[:batch_size]
    X12 = X_imputed[:SITE1_CELL + SITE2_CELL][indices1]
    X3  = X_imputed[-SITE3_CELL:][indices2]
    GEX = torch.transpose(X_imputed[:, :FILLED_GEX], 0, 1)
    ADT = torch.transpose(X_imputed[:, FILLED_GEX:], 0, 1)

    _, _, C1 = dkm(X12)
    _, _, C2 = dkm(X3)

    M = F.normalize(torch.cdist(C1, C2))
    P = tools.gumbel_sinkhorn(M)
    h_loss = (M * P).sum()
    loss = (0.5 * ot.sliced_wasserstein_distance(X12, X3, n_projections=n_projections) +
            0.5 * ot.sliced_wasserstein_distance(GEX, ADT, n_projections=n_projections) +
            0.1 * h_loss)
    print(f"{epoch}: h_loss = {h_loss.item():.4f}, loss = {loss.item():.4f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 300 == 0:
        X_imputed = X.detach().clone()
        X_imputed[mask] = imps

        pearson_corr = pearsonr(X_imputed[-SITE3_CELL:, :FILLED_GEX][nonzero_mask31].detach().cpu().numpy(), ground_truth[-SITE3_CELL:, :FILLED_GEX][nonzero_mask31].detach().cpu().numpy())[0]
        citeseq.X = np.vstack((X_imputed.detach().cpu().numpy(), X4))
        ari, nmi = tools.clustering(citeseq)
        print(f"Iteration {epoch + 1}/{epochs}: loss: {loss.item():.4f}, pearson: {pearson_corr:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}")
        wandb.log({"Iteration": epoch + 1, "loss": loss, "pearson": pearson_corr, "ari": ari, "nmi": nmi})
