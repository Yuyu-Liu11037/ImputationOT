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
import argparse
from scipy.stats import pearsonr
from geomloss import SamplesLoss
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from utils import tools

parser = argparse.ArgumentParser()
parser.add_argument("--use_wandb", type=bool, default=False)
parser.add_argument("--aux_weight", type=float, default=1)
parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--eval_interval", type=int, default=100)
parser.add_argument("--start_aux", type=int, default=400)
parser.add_argument("--batch_size", type=int, default=3000)
parser.add_argument("--seed", type=int, default=2024)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:0'
SITE1_CELL = 16311
SITE2_CELL = 25171
SITE3_CELL = 32029
SITE4_CELL = 16750
FILLED_GEX = 2000

if args.use_wandb is True:
    wandb.init(
        project="ot",
        group="citeseq-4adt", 
        job_type="aux",
        name="SamplesLoss+fixed_h_loss",
        config={
            "dataset": "NIPS2021-Cite-seq",
            "epochs": args.epochs,
            "missing data": "site4 adt",
            "h_loss weight": args.aux_weight,
            "comment": "nothing"
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
citeseq = ad.concat([adata_GEX, adata_ADT], axis=1, merge="first")  # X(:,1): GEX, X(:,2): ADT
print(f"Finish preprocessing\n")
#####################################################################################################################################

X = citeseq.X.toarray()
X3 = X[SITE1_CELL + SITE2_CELL:SITE1_CELL + SITE2_CELL + SITE3_CELL].copy()
X = torch.tensor(X).to(device)
X = torch.cat((X[:SITE1_CELL + SITE2_CELL], X[-SITE4_CELL:]), dim=0)  # Matrix is too large. Remove certain rows to save memory.
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

optimizer = optim.Adam([imps])
lambda_lr = lambda epoch: 0.1 if epoch < 200 else 0.001
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

h_loss = torch.zeros(1).to(device)

print("Start optimizing")
for epoch in range(args.epochs):
    X_imputed = X.detach().clone()
    X_imputed[mask] = imps

    if epoch == 0 and args.use_wandb is True:
        pearson_corr = pearsonr(X_imputed[-SITE4_CELL:, 2000:][nonzero_mask42].detach().cpu().numpy(), ground_truth[-SITE4_CELL:, 2000:][nonzero_mask42].detach().cpu().numpy())[0]
        citeseq.X = np.vstack((X_imputed[:SITE1_CELL + SITE2_CELL].detach().cpu().numpy(), X3, X_imputed[-SITE4_CELL:].detach().cpu().numpy()))
        ari, nmi, _ = tools.clustering(citeseq)
        print(f"Initial pearson: {pearson_corr:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}")
        wandb.log({"Iteration": epoch, "loss": 0, "pearson": pearson_corr, "ari": ari, "nmi": nmi})

    
    indices1 = torch.randperm(SITE1_CELL + SITE2_CELL, device=device)[:args.batch_size]
    indices2 = torch.randperm(SITE4_CELL, device=device)[:args.batch_size]
    X12 = X_imputed[:SITE1_CELL + SITE2_CELL][indices1]
    X4  = X_imputed[-SITE4_CELL:][indices2]
    GEX = torch.transpose(X_imputed[:, :2000], 0, 1)
    ADT = torch.transpose(X_imputed[:, 2000:], 0, 1)

    if epoch >= args.start_aux:
        if epoch % args.eval_interval == 0:
            ### calculate cluster results
            labels1 = tools.calculate_cluster_labels(X12)
            labels2 = tools.calculate_cluster_labels(X4)
        ### calculate cluster centroids
        centroids1 = tools.calculate_cluster_centroids(X12, labels1)
        centroids2 = tools.calculate_cluster_centroids(X4, labels2)
        ### calculate cluster loss
        M = torch.cdist(centroids1, centroids2)
        P = tools.gumbel_sinkhorn(M, tau=1, n_iter=5)
        h_loss = (M * P).sum()
        # h_loss = nn.CrossEntropyLoss()(centroids1, centroids2)
    
    w_h = 0 if epoch < args.start_aux else args.aux_weight
    omics_loss = SamplesLoss()(GEX, ADT)
    cells_loss = SamplesLoss()(X12, X4)
    loss = 0.5 * 0.001 * omics_loss + 0.5 * cells_loss + w_h * h_loss
    print(f"{epoch}: omics_loss = {omics_loss.item():.4f}, cells_loss = {cells_loss.item():.4f}, h_loss = {h_loss.item():.4f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % args.eval_interval == 0 and args.use_wandb is True:
        X_imputed = X.detach().clone()
        X_imputed[mask] = imps
        
        pearson_corr = pearsonr(X_imputed[-SITE4_CELL:, 2000:][nonzero_mask42].detach().cpu().numpy(), ground_truth[-SITE4_CELL:, 2000:][nonzero_mask42].detach().cpu().numpy())[0]
        citeseq.X = np.vstack((X_imputed[:SITE1_CELL + SITE2_CELL].detach().cpu().numpy(), X3, X_imputed[-SITE4_CELL:].detach().cpu().numpy()))
        ari, nmi, _ = tools.clustering(citeseq)
        print(f"Iteration {epoch + 1}/{args.epochs}: loss: {loss.item():.4f}, pearson: {pearson_corr:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}")
        wandb.log({"Iteration": epoch + 1, "loss": loss, "pearson": pearson_corr, "ari": ari, "nmi": nmi})
