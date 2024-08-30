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
parser.add_argument("--use_wandb", action="store_true", default=False)
parser.add_argument("--aux_weight", type=float, default=1.0)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--eval_interval", type=int, default=25)
parser.add_argument("--start_aux", type=int, default=400)
parser.add_argument("--batch_size", type=int, default=3000)
parser.add_argument("--seed", type=int, default=2024)

parser.add_argument("--wandb_group", type=str, default="multiome-4gex")
parser.add_argument("--wandb_job", type=str, choices=["main", "ablation", "aux"], default="main")
parser.add_argument("--wandb_name", type=str, default="exp")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:0'
SITE1 = 17243
SITE2 = 15226
SITE3 = 14556
SITE4 = 22224

if args.use_wandb is True:
    wandb.init(
        project="ot",
        group=args.wandb_group,
        job_type=args.wandb_job,
        name=args.wandb_name,
        config={
            "dataset": "NIPS2021-Multiome",
            "epochs": args.epochs,
            "h_loss weight": args.aux_weight,
            "comment": "use counts layer"
        }
    )

multiome = ad.read_h5ad("/workspace/ImputationOT/data/multiome_processed.h5ad")   # 22 cell types
multiome.var_names_make_unique()

#####################################################################################################################################
print("Start preprocessing")
### preprocess
adata_GEX = multiome[:, multiome.var['feature_types'] == 'GEX'].copy()
adata_ATAC = multiome[:, multiome.var['feature_types'] == 'ATAC'].copy()
### step 1: normalize
print("Use normalization")
sc.pp.normalize_total(adata_GEX, target_sum=1e4, layer='counts')
sc.pp.normalize_total(adata_ATAC, target_sum=1e4, layer='counts')
### step 2: log transform
sc.pp.log1p(adata_GEX, layer='counts')
sc.pp.log1p(adata_ATAC, layer='counts')
### step 3: select highly variable features
sc.pp.highly_variable_genes(adata_GEX, subset=True, layer='counts')
sc.pp.highly_variable_genes(
    adata_ATAC,
    n_top_genes=4000,
    subset=True, 
    layer='counts'
)

num_atac = adata_ATAC.layers['counts'].shape[1]
multiome = ad.concat([adata_ATAC, adata_GEX], axis=1, merge="first")   # left num_atac: ATAC, right 2832: GEX
print(f"Finish preprocessing\n")
#####################################################################################################################################

X = multiome.layers['counts'].toarray()
X12 = X[:SITE1 + SITE2]
X = torch.tensor(X).to(device)
X = X[SITE1 + SITE2:]   # Matrix is too large. Remove certain rows to save memory.
ground_truth = X.clone()

mask = torch.zeros(X.shape, dtype=torch.bool).to(device)
mask[-SITE4:, num_atac:] = True   # mask X(4,2)
nonzero_mask32 = (X[:SITE3, num_atac:] != 0).to(device)   # nonzero data of X(3,2)
nonzero_mask41 = (X[-SITE4:, :num_atac] != 0).to(device)   # nonzero data of X(4,1)
nonzero_mask42 = (X[-SITE4:, num_atac:] != 0).to(device)   # nonzero data of X(4,2)

mean_values = torch.sum(X[:SITE3, num_atac:], dim=0) / torch.sum(nonzero_mask32, dim=0)
imps = mean_values.repeat(SITE4).to(device)
imps += torch.randn(imps.shape, device=device) * 0.1
imps.requires_grad = True

def lr_lambda(epoch):
    if epoch < 20:
        return 1.0
    elif 20 <= epoch < 100:
        return 1.1 - (epoch - 20) / 80.0
    elif 100 <= epoch < 1000:
        return 0.101 - (epoch - 100) / 9000.0
    else:
        return 0.001

optimizer = optim.Adam([imps], 1.0)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

h_loss = torch.zeros(1).to(device)
omics_loss = torch.zeros(1).to(device)
cells_loss = torch.zeros(1).to(device)

print("Start optimizing")
for epoch in range(args.epochs):
    X_imputed = X.detach().clone()
    X_imputed[mask] = imps

    if epoch == 0 and args.use_wandb is True:
        pearson_corr = pearsonr(X_imputed[-SITE4:, num_atac:][nonzero_mask42].detach().cpu().numpy(), ground_truth[-SITE4:, num_atac:][nonzero_mask42].detach().cpu().numpy())[0]
        multiome.X = np.vstack((X12, X_imputed.detach().cpu().numpy()))
        ari, nmi = tools.cluster_with_kmeans(multiome, n_clusters=20, use_pca=False)
        print(f"Initial pearson: {pearson_corr:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}")
        wandb.log({"Iteration": 0, "loss": 0, "pearson": pearson_corr, "ari": ari, "nmi": nmi})

    indices1 = torch.randperm(SITE3, device=device)[:args.batch_size]
    indices2 = torch.randperm(SITE4, device=device)[:args.batch_size]
    X3 = X_imputed[:SITE3][indices1]
    X4  = X_imputed[-SITE4:][indices2]
    # ATAC = torch.transpose(X_imputed[:, :num_atac], 0, 1)
    # GEX  = torch.transpose(X_imputed[:, num_atac:], 0, 1)

    # if epoch >= args.start_aux:
    #     if epoch % args.eval_interval == 0:
    #         ### calculate cluster results
    #         labels1 = tools.calculate_cluster_labels(X12, resolution=0.65)
    #         labels2 = tools.calculate_cluster_labels(X4, resolution=0.65)
    #     ### calculate cluster centroids
    #     centroids1 = tools.calculate_cluster_centroids(X12, labels1)
    #     centroids2 = tools.calculate_cluster_centroids(X4, labels2)
    #     ### calculate cluster loss
    #     M = torch.cdist(centroids1, centroids2)
    #     P = tools.gumbel_sinkhorn(M, tau=1, n_iter=5)
    #     h_loss = (M * P).sum()
    
    w_h = 0 if epoch < args.start_aux else args.aux_weight
    # omics_loss = SamplesLoss()(GEX, ATAC)
    cells_loss = SamplesLoss()(X3, X4)
    # loss = 0.5 * 0.1 * omics_loss + 0.5 * 0.1 * cells_loss + w_h * h_loss
    loss = cells_loss
    print(f"{epoch}: omics_loss = {omics_loss.item():.4f}, cells_loss = {cells_loss.item():.4f}, h_loss = {h_loss.item():.4f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % args.eval_interval == 0 and args.use_wandb is True:
        X_imputed = X.detach().clone()
        X_imputed[mask] = imps
        
        pearson_corr = pearsonr(X_imputed[-SITE4:, num_atac:][nonzero_mask42].detach().cpu().numpy(), ground_truth[-SITE4:, num_atac:][nonzero_mask42].detach().cpu().numpy())[0]
        multiome.X = np.vstack((X12, X_imputed.detach().cpu().numpy()))
        ari, nmi = tools.cluster_with_leiden(multiome, resolution_values=[0.15, 0.20, 0.25])
        print(f"Iteration {epoch + 1}/{args.epochs}: loss: {loss.item():.4f}, pearson: {pearson_corr:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}")
        wandb.log({"Iteration": epoch + 1, "loss": loss, "pearson": pearson_corr, "ari": ari, "nmi": nmi})
