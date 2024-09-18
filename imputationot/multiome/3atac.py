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
parser.add_argument("--resolution_values", type=str_to_float_list, default=[0.55, 0.65, 0.75, 0.85])

parser.add_argument("--wandb_group", type=str, default="multiome-3atac")
parser.add_argument("--wandb_job", type=str, choices=["main", "ablation", "aux"], default="ablation")
parser.add_argument("--wandb_name", type=str, default="cells")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:0'
SITE1_CELL = 17243
SITE2_CELL = 15226
SITE3_CELL = 14556
SITE4_CELL = 22224

if args.use_wandb is True:
    wandb.init(
        project="ot",
        group=args.wandb_group,
        job_type=args.wandb_job,
        name=args.wandb_name
    )

multiome = ad.read_h5ad("/workspace/ImputationOT/imputationot/data/multiome_processed.h5ad")
multiome.var_names_make_unique()

#####################################################################################################################################
### preprocess
adata_GEX = multiome[:, multiome.var['feature_types'] == 'GEX'].copy()
adata_ATAC = multiome[:, multiome.var['feature_types'] == 'ATAC'].copy()
### step 1: normalize
sc.pp.normalize_total(adata_GEX, target_sum=1e4)
sc.pp.normalize_total(adata_ATAC, target_sum=1e4)
### step 2: log transform
sc.pp.log1p(adata_GEX)
sc.pp.log1p(adata_ATAC)
### step 3: select highly variable features
sc.pp.highly_variable_genes(adata_GEX, subset=True)
sc.pp.highly_variable_genes(
    adata_ATAC,
    n_top_genes=4000,
    subset=True
)

num_atac = adata_ATAC.X.shape[1]
multiome = ad.concat([adata_ATAC, adata_GEX], axis=1, merge="first")   # left num_atac: ATAC, right 2832: GEX
print(f"Finish preprocessing\n")
#####################################################################################################################################

X = multiome.X.toarray()
X4 = X[SITE1_CELL + SITE2_CELL + SITE3_CELL:].copy()
X = torch.tensor(X).to(device)
X = X[:SITE1_CELL + SITE2_CELL + SITE3_CELL]   # Matrix is too large. Remove certain rows to save memory.
ground_truth = X.clone()

mask = torch.zeros(X.shape, dtype=torch.bool).to(device)
mask[SITE1_CELL + SITE2_CELL: SITE1_CELL + SITE2_CELL + SITE3_CELL, :num_atac] = True   # mask X(3,1)

nonzero_mask1121 = (X[:SITE1_CELL + SITE2_CELL, :num_atac] != 0).to(device)   # nonzero data of X(1,1), X(2,1)
nonzero_mask31 = (X[-SITE3_CELL:, :num_atac] != 0).to(device)   # nonzero data of X(3,1)
nonzero_mask32 = (X[-SITE3_CELL:, num_atac:] != 0).to(device)   # nonzero data of X(3,2)
mean_values = torch.sum(X[:SITE1_CELL + SITE2_CELL, :num_atac], dim=0) / torch.sum(nonzero_mask1121, dim=0)
imps = mean_values.repeat(SITE3_CELL).to(device)
imps += torch.randn(imps.shape, device=device) * 0.1
imps.requires_grad = True

optimizer = optim.Adam([imps], 0.1)

h_loss = torch.zeros(1).to(device)
cells_loss = torch.zeros(1).to(device)
lr = 0.1

print("Start optimizing")
for epoch in range(args.epochs):
    optimizer.zero_grad()
    X_imputed = X.detach().clone()
    X_imputed[mask] = imps

    if epoch == 0 and args.use_wandb is True:
        ### pearson
        pearson_corr = pearsonr(X_imputed[-SITE3_CELL:, :num_atac][nonzero_mask31].detach().cpu().numpy(), ground_truth[-SITE3_CELL:, :num_atac][nonzero_mask31].detach().cpu().numpy())[0]
        ### mae & rmse
        mae, rmse = calculate_mae_rmse(X_imputed[-SITE3_CELL:, :num_atac], ground_truth[-SITE3_CELL:, :num_atac], nonzero_mask31)
        ### cmd
        cmd = correlation_matrix_distance(
            correlation_matrix(X_imputed[-SITE3_CELL:, :num_atac].detach().cpu()), correlation_matrix(ground_truth[-SITE3_CELL:, :num_atac].detach().cpu()))
        ### ari & nmi & purity & jaccard
        multiome.X = np.vstack((X_imputed.detach().cpu().numpy(), X4))
        ari, nmi, purity, jaccard = cluster_with_leiden(multiome, resolution_values=args.resolution_values)
        print(f"Initial pearson: {pearson_corr:.4f}, mae: {mae:.4f}, rmse: {rmse:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}, purity: {purity:.4f}, jaccard: {jaccard:.4f}")
        wandb.log({"Iteration": 0, "loss": 0, "pearson": pearson_corr, "mae": mae, "rmse": rmse, "ari": ari, "nmi": nmi, "purity": purity})

    indices1 = torch.randperm(SITE1_CELL + SITE2_CELL, device=device)[:args.batch_size]
    indices2 = torch.randperm(SITE3_CELL, device=device)[:args.batch_size]
    X12 = X_imputed[:SITE1_CELL + SITE2_CELL][indices1]
    X3  = X_imputed[-SITE3_CELL:][indices2]

    if args.use_cluster is True:
        labels1 = calculate_cluster_labels(X12, resolution=0.65)
        labels2 = calculate_cluster_labels(X3, resolution=0.65)
        ### calculate cluster centroids
        centroids1 = calculate_cluster_centroids(X12, labels1)
        centroids2 = calculate_cluster_centroids(X3, labels2)
        ### calculate cluster loss
        h_loss = SamplesLoss()(centroids1, centroids2)
    
    cells_loss = SamplesLoss()(X12, X3)
    loss = args.weights[0] * cells_loss + args.weights[1] * h_loss
    print(f"{epoch}: lr = {lr}, cells_loss = {cells_loss.item():.4f}, h_loss = {h_loss.item():.4f}")
    
    loss.backward()
    optimizer.step()

    if (epoch + 1) % args.eval_interval == 0 and args.use_wandb is True:
        X_imputed = X.detach().clone()
        X_imputed[mask] = imps

        ### pearson
        pearson_corr = pearsonr(X_imputed[-SITE3_CELL:, :num_atac][nonzero_mask31].detach().cpu().numpy(), ground_truth[-SITE3_CELL:, :num_atac][nonzero_mask31].detach().cpu().numpy())[0]
        ### mae & rmse
        mae, rmse = calculate_mae_rmse(X_imputed[-SITE3_CELL:, :num_atac], ground_truth[-SITE3_CELL:, :num_atac], nonzero_mask31)
        ### cmd
        cmd = correlation_matrix_distance(
            correlation_matrix(X_imputed[-SITE3_CELL:, :num_atac].detach().cpu()), correlation_matrix(ground_truth[-SITE3_CELL:, :num_atac].detach().cpu()))
        ### ari & nmi & purity & jaccard
        multiome.X = np.vstack((X_imputed.detach().cpu().numpy(), X4))
        ari, nmi, purity, jaccard = cluster_with_leiden(multiome, resolution_values=args.resolution_values)
        print(f"pearson: {pearson_corr:.4f}, mae: {mae:.4f}, rmse: {rmse:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}, purity: {purity:.4f}, jaccard: {jaccard:.4f}")
        wandb.log({"Iteration": epoch + 1, "loss": loss, "pearson": pearson_corr, "mae": mae, "rmse": rmse, "ari": ari, "nmi": nmi, "purity": purity})
