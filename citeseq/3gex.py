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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import tools
from weighting.MGDA import MGDA

parser = argparse.ArgumentParser()
parser.add_argument("--use_wandb", action="store_true", default=False)
parser.add_argument("--aux_weight", type=float, default=0)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--eval_interval", type=int, default=5)
parser.add_argument("--start_aux", type=int, default=40)
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
        group="citeseq-3gex", 
        job_type="aux",
        name="mgda",
        config={
            "dataset": "NIPS2021-Cite-seq",
            "epochs": args.epochs,
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

optimizer = optim.Adam([imps], 0.1)
mgda = MGDA()
mgda_gn = 'none'

h_loss = torch.zeros(1).to(device)
omics_loss = torch.zeros(1).to(device)
cells_loss = torch.zeros(1).to(device)
lr = 0.1

print("Start optimizing")
for epoch in range(args.epochs):
    X_imputed = X.detach().clone()
    X_imputed[mask] = imps

    if epoch == 0 and args.use_wandb is True:
        ### pearson
        pearson_corr = pearsonr(X_imputed[-SITE3_CELL:, :FILLED_GEX][nonzero_mask31].detach().cpu().numpy(), ground_truth[-SITE3_CELL:, :FILLED_GEX][nonzero_mask31].detach().cpu().numpy())[0]
        ### mse
        mse = F.mse_loss(X_imputed[-SITE3_CELL:, :FILLED_GEX].detach().cpu(), ground_truth[-SITE3_CELL:, :FILLED_GEX].detach().cpu())
        ### cmd
        cmd = tools.correlation_matrix_distance(tools.correlation_matrix(X_imputed[-SITE3_CELL:, :FILLED_GEX].detach().cpu()), tools.correlation_matrix(ground_truth[-SITE3_CELL:, :FILLED_GEX].detach().cpu()))
        ### ari & nmi
        citeseq.X = np.vstack((X_imputed.detach().cpu().numpy(), X4))
        ari, nmi = tools.clustering(citeseq)
        
        print(f"Iteration {epoch + 1}/{args.epochs}: pearson: {pearson_corr:.4f}, mse: {mse:.4f}, cmd: {cmd:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}")
        wandb.log({"Iteration": epoch, "loss": 0, "pearson": pearson_corr, "mse": mse, "cmd": cmd, "ari": ari, "nmi": nmi})

    indices1 = torch.randperm(SITE1_CELL + SITE2_CELL, device=device)[:args.batch_size]
    indices2 = torch.randperm(SITE3_CELL, device=device)[:args.batch_size]
    X12 = X_imputed[:SITE1_CELL + SITE2_CELL][indices1]
    X3  = X_imputed[-SITE3_CELL:][indices2]

    if epoch >= 1:
        labels1 = tools.calculate_cluster_labels(X12)
        labels2 = tools.calculate_cluster_labels(X3)
        ### calculate cluster centroids
        centroids1 = tools.calculate_cluster_centroids(X12, labels1)
        centroids2 = tools.calculate_cluster_centroids(X3, labels2)
        ### calculate cluster loss
        h_loss = SamplesLoss()(centroids1, centroids2)
    
    cells_loss = SamplesLoss()(X12, X3)
    losses = [cells_loss, h_loss]
    sol = mgda.backward(losses, mgda_gn=mgda_gn)
    print(sol)
    print(f"{epoch}: lr = {lr}, omics_loss = {omics_loss.item():.4f}, cells_loss = {cells_loss.item():.4f}, h_loss = {h_loss.item():.4f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % args.eval_interval == 0 and args.use_wandb is True:
        X_imputed = X.detach().clone()
        X_imputed[mask] = imps

        ### pearson
        pearson_corr = pearsonr(X_imputed[-SITE3_CELL:, :FILLED_GEX][nonzero_mask31].detach().cpu().numpy(), ground_truth[-SITE3_CELL:, :FILLED_GEX][nonzero_mask31].detach().cpu().numpy())[0]
        ### mse
        mse = F.mse_loss(X_imputed[-SITE3_CELL:, :FILLED_GEX].detach().cpu(), ground_truth[-SITE3_CELL:, :FILLED_GEX].detach().cpu())
        ### cmd
        cmd = tools.correlation_matrix_distance(tools.correlation_matrix(X_imputed[-SITE3_CELL:, :FILLED_GEX].detach().cpu()), tools.correlation_matrix(ground_truth[-SITE3_CELL:, :FILLED_GEX].detach().cpu()))
        ### ari & nmi
        citeseq.X = np.vstack((X_imputed.detach().cpu().numpy(), X4))
        ari, nmi = tools.clustering(citeseq)
        
        print(f"Iteration {epoch + 1}/{args.epochs}: loss: {loss.item():.4f}, pearson: {pearson_corr:.4f}, mse: {mse:.4f}, cmd: {cmd:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}")
        wandb.log({"Iteration": epoch + 1, "loss": loss, "pearson": pearson_corr, "mse": mse, "cmd": cmd, "ari": ari, "nmi": nmi})
