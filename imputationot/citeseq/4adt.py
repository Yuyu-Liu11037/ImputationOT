import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import anndata as ad
import scanpy as sc
import wandb
import sys
import os
import random
import argparse
import time
from scipy.stats import pearsonr
from geomloss import SamplesLoss

from imputationot.utils import correlation_matrix, correlation_matrix_distance, calculate_mae_rmse, calculate_cluster_labels, calculate_cluster_centroids, cluster_with_leiden

def str_to_float_list(arg):
    return [float(x) for x in arg.split(',')]

parser = argparse.ArgumentParser()
parser.add_argument("--use_wandb", action="store_true", default=False)
parser.add_argument("--use_cluster", action="store_true", default=False)
parser.add_argument("--weights", type=str_to_float_list, default=[1,0])
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--eval_interval", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=3000)
parser.add_argument("--seed", type=int, default=2024)
parser.add_argument("--zero_percent", type=int, default=50)
parser.add_argument("--resolution_values", type=str_to_float_list, default=[0.15, 0.20, 0.25])

parser.add_argument("--wandb_group", type=str, default="citeseq-4adt")
parser.add_argument("--wandb_job", type=str, choices=["main", "ablation", "aux"], default="aux")
parser.add_argument("--wandb_name", type=str, default="equal_weighting")
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
        group=args.wandb_group,
        job_type=args.wandb_job,
        name=args.wandb_name
    )

citeseq = ad.read_h5ad("/workspace/ImputationOT/imputationot/data/citeseq_processed.h5ad")
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

# def lr_lambda(epoch):
#     if epoch < 10:
#         return 0.1
#     elif 300 <= epoch < 1000:
#         return 0.101 - (epoch - 300) / 7000.0
#     else:
#         return 0.001

# optimizer = optim.Adam([imps], 1.0)
optimizer = optim.Adam([imps], 0.1)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

h_loss = torch.zeros(1).to(device)
cells_loss = torch.zeros(1).to(device)
lr = 0.1

print("Start optimizing")
start_time = time.time()
for epoch in range(args.epochs):
    optimizer.zero_grad()
    X_imputed = X.detach().clone()
    X_imputed[mask] = imps

    if epoch == 0 and args.use_wandb is True:
        ### pearson
        pearson_corr = pearsonr(X_imputed[-SITE4_CELL:, 2000:][nonzero_mask42].detach().cpu().numpy(), ground_truth[-SITE4_CELL:, 2000:][nonzero_mask42].detach().cpu().numpy())[0]
        ### mae & rmse
        mae, rmse = calculate_mae_rmse(X_imputed[-SITE4_CELL:, 2000:], ground_truth[-SITE4_CELL:, 2000:], nonzero_mask42)
        ### cmd
        cmd = correlation_matrix_distance(correlation_matrix(X_imputed[-SITE4_CELL:, 2000:].detach().cpu()), correlation_matrix(ground_truth[-SITE4_CELL:, 2000:].detach().cpu()))
        ### ari & nmi & purity & jaccard
        citeseq.X = np.vstack((X_imputed[:SITE1_CELL + SITE2_CELL].detach().cpu().numpy(), X3, X_imputed[-SITE4_CELL:].detach().cpu().numpy()))
        ari, nmi, purity, jaccard = cluster_with_leiden(citeseq, resolution_values=args.resolution_values)
        print(f"Initial pearson: {pearson_corr:.4f}, mae: {mae:.4f}, rmse: {rmse:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}, purity: {purity:.4f}, jaccard: {jaccard:.4f}")
        wandb.log({"Iteration": epoch + 1, "loss": 0, "pearson": pearson_corr, "mae": mae, "rmse": rmse, "ari": ari, "nmi": nmi, "purity": purity})

    
    indices1 = torch.randperm(SITE1_CELL + SITE2_CELL, device=device)[:args.batch_size]
    indices2 = torch.randperm(SITE4_CELL, device=device)[:args.batch_size]
    X12 = X_imputed[:SITE1_CELL + SITE2_CELL][indices1]
    X4  = X_imputed[-SITE4_CELL:][indices2]

    if args.use_cluster is True:
        labels1 = calculate_cluster_labels(X12)
        labels2 = calculate_cluster_labels(X4)
        ### calculate cluster centroids
        centroids1 = calculate_cluster_centroids(X12, labels1)
        centroids2 = calculate_cluster_centroids(X4, labels2)
        ### calculate cluster loss
        h_loss = SamplesLoss()(centroids1, centroids2)
    
    cells_loss = SamplesLoss()(X12, X4)
    loss = args.weights[0] * cells_loss + args.weights[1] * h_loss
    print(f"{epoch}: cells_loss = {cells_loss.item():.4f}, h_loss = {h_loss.item():.4f}")

    loss.backward()
    optimizer.step()
    # scheduler.step()

    if (epoch + 1) % args.eval_interval == 0 and args.use_wandb is True:
        X_imputed = X.detach().clone()
        X_imputed[mask] = imps
        
        ### pearson
        pearson_corr = pearsonr(X_imputed[-SITE4_CELL:, 2000:][nonzero_mask42].detach().cpu().numpy(), ground_truth[-SITE4_CELL:, 2000:][nonzero_mask42].detach().cpu().numpy())[0]
        ### mae & rmse
        mae, rmse = calculate_mae_rmse(X_imputed[-SITE4_CELL:, 2000:], ground_truth[-SITE4_CELL:, 2000:], nonzero_mask42)
        ### cmd
        cmd = correlation_matrix_distance(correlation_matrix(X_imputed[-SITE4_CELL:, 2000:].detach().cpu()), correlation_matrix(ground_truth[-SITE4_CELL:, 2000:].detach().cpu()))
        ### ari & nmi & purity & jaccard
        citeseq.X = np.vstack((X_imputed[:SITE1_CELL + SITE2_CELL].detach().cpu().numpy(), X3, X_imputed[-SITE4_CELL:].detach().cpu().numpy()))
        ari, nmi, purity, jaccard = cluster_with_leiden(citeseq, resolution_values=args.resolution_values)
        print(f"Initial pearson: {pearson_corr:.4f}, mae: {mae:.4f}, rmse: {rmse:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}, purity: {purity:.4f}, jaccard: {jaccard:.4f}")
        wandb.log({"Iteration": epoch + 1, "loss": loss, "pearson": pearson_corr, "mae": mae, "rmse": rmse, "ari": ari, "nmi": nmi, "purity": purity})
print(f"Time usage: {time.time() - start_time:.2f}")
