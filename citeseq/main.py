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
        project="general imputation",
        group="citeseq-4adt", 
        job_type="main",
        name="SamplesLoss-5k"
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
X = torch.tensor(X).to(device)

ari, nmi = tools.clustering(citeseq)
print(f"Initial ari: {ari:.4f}, nmi: {nmi:.4f}")
wandb.log({"Iteration": 0, "loss": 0, "ari": ari, "nmi": nmi})

mask = torch.zeros(X.shape, dtype=torch.bool).to(device)
mask[X == 0] = True

col_means = torch.zeros(X.shape[1]).to(device)
for col in range(X.shape[1]):
    non_zero_values = X[:, col][X[:, col] != 0]
    if len(non_zero_values) > 0:
        col_means[col] = non_zero_values.mean()

imps_init = torch.zeros_like(X).to(device)
for col in range(X.shape[1]):
    imps_init[:, col][mask[:, col]] = col_means[col]

imps = torch.nn.Parameter(imps_init[mask])

def lr_lambda(epoch):
    if epoch < 300:
        return 0.1
    elif 300 <= epoch < 1000:
        return 0.101 - (epoch - 300) / 7000.0
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
    
    indices = torch.randperm(SITE1_CELL + SITE2_CELL + SITE3_CELL + SITE4_CELL, device=device)
    indices1 = indices[:args.batch_size]
    indices2 = indices[args.batch_size:args.batch_size * 2]
    X1 = X_imputed[indices1]
    X2 = X_imputed[indices2]
    
    w_h = 0 if epoch < args.start_aux else args.aux_weight
    # omics_loss = SamplesLoss()(GEX, ADT)
    cells_loss = SamplesLoss()(X1, X2)
    # loss = 0.5 * 0.001 * omics_loss + 0.5 * cells_loss + w_h * h_loss
    loss = cells_loss
    print(f"{epoch}: omics_loss = {omics_loss.item():.4f}, cells_loss = {cells_loss.item():.4f}, h_loss = {h_loss.item():.4f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % args.eval_interval == 0 and args.use_wandb is True:
        X_imputed = X.detach().clone()
        X_imputed[mask] = imps
        
        citeseq.X = X_imputed.detach().cpu().numpy()
        ari, nmi = tools.clustering(citeseq)
        print(f"Iteration {epoch + 1}/{args.epochs}: loss: {loss.item():.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}")
        wandb.log({"Iteration": epoch + 1, "loss": loss, "ari": ari, "nmi": nmi})
