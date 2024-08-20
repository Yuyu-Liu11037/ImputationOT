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

parser.add_argument("--source_batches", type=str, default="1,2", help="Impute batch size range from 1 to 9")
parser.add_argument("--target_batch", type=int, default=3, help="Batch number to impute")

parser.add_argument("--wandb_group", type=str, default="pbmc")
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
batch_sizes = [13086, 8618, 8243, 10553, 11017, 11170, 11752, 12745, 10315]
batch_sizes_cumsum = np.cumsum([0] + batch_sizes)

source_batches = list(map(int, args.source_batches.split(',')))
target_batch = args.target_batch
remaining_batches = [i for i in range(1, len(batch_sizes) + 1) if i not in source_batches and i != target_batch]

if args.use_wandb is True:
    wandb.init(
        project="ot",
        group=args.wandb_group,
        job_type=args.wandb_job,
        name=args.wandb_name,
        config={
            "dataset": "pbmc",
            "epochs": args.epochs,
            "h_loss weight": args.aux_weight
        }
    )

pbmc = ad.read_h5ad("/workspace/ImputationOT/data/vento_pbmc_processed.h5ad") 
pbmc.var_names_make_unique()

#####################################################################################################################################
print("Start preprocessing")
### preprocess
### step 1: normalize
sc.pp.normalize_total(pbmc, target_sum=1e4)
### step 2: log transform
sc.pp.log1p(pbmc)
### step 3: select highly variable features
sc.pp.highly_variable_genes(pbmc, subset=True)
print(f"Finish preprocessing\n")
#####################################################################################################################################

X = pbmc.X.toarray()
X = torch.tensor(X).to(device)
X_source = torch.cat([X[batch_sizes_cumsum[i - 1]:batch_sizes_cumsum[i]] for i in source_batches], dim=0)
X_target = X[batch_sizes_cumsum[target_batch - 1]:batch_sizes_cumsum[target_batch]]
X_del = torch.cat([X[batch_sizes_cumsum[i-1]:batch_sizes_cumsum[i]] for i in remaining_batches], dim=0) 

ground_truth = X_target.clone()

mask = torch.zeros(X_target.shape, dtype=torch.bool).to(device)
nonzero_mask_source = (X_source != 0).to(device)
nonzero_mask_target = (X_target != 0).to(device) 

epsilon = 1e-8
mean_values = torch.sum(X_source * nonzero_mask_source, dim=0) / (torch.sum(nonzero_mask_source, dim=0) + epsilon)
imps = mean_values.repeat(X_target.shape[0]).to(device)
imps += torch.randn(imps.shape, device=device) * 0.1
imps.requires_grad = True
imps = imps.view(X_target.shape).detach().requires_grad_()

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
cells_loss = torch.zeros(1).to(device)

print(f"Start optimizing: use batch(es) {args.source_batches} to impute batch {args.target_batch}")
for epoch in range(args.epochs):
    X_imputed = imps

    if epoch == 0 and args.use_wandb is True:
        pearson_corr = pearsonr(X_imputed[nonzero_mask_target].detach().cpu().numpy(), ground_truth[nonzero_mask_target].detach().cpu().numpy())[0]
        X_full = []
        for i in range(1, len(batch_sizes) + 1):
            if i in source_batches:
                tmp = X_source[batch_sizes_cumsum[i-1]:batch_sizes_cumsum[i]].detach().cpu().numpy()
            elif i == target_batch:
                tmp = X_imputed.detach().cpu().numpy()
            else:
                tmp = X[batch_sizes_cumsum[i-1]:batch_sizes_cumsum[i]].detach().cpu().numpy()
            X_full.append(tmp)
        pbmc.X = np.vstack(X_full)
        ari, nmi = tools.cluster_with_kmeans(pbmc, n_clusters=20, use_pca=False)
        print(f"Initial pearson: {pearson_corr:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}")
        wandb.log({"Iteration": 0, "loss": 0, "pearson": pearson_corr, "ari": ari, "nmi": nmi})

    indices1 = torch.randperm(X_source.shape[0], device=device)[:args.batch_size]
    indices2 = torch.randperm(X_target.shape[0], device=device)[:args.batch_size]
    X1 = X_source[indices1]
    X2 = X_imputed[indices2]
    
    w_h = 0 if epoch < args.start_aux else args.aux_weight
    cells_loss = SamplesLoss()(X1, X2)
    loss = cells_loss
    print(f"{epoch}: cells_loss = {cells_loss.item():.4f}, h_loss = {h_loss.item():.4f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % args.eval_interval == 0 and args.use_wandb is True:
        X_imputed = imps
        
        pearson_corr = pearsonr(X_imputed[nonzero_mask_target].detach().cpu().numpy(), ground_truth[nonzero_mask_target].detach().cpu().numpy())[0]
        X_full = []
        for i in range(1, len(batch_sizes) + 1):
            if i in source_batches:
                tmp = X_source[batch_sizes_cumsum[i-1]:batch_sizes_cumsum[i]].detach().cpu().numpy()
            elif i == target_batch:
                tmp = X_imputed.detach().cpu().numpy()
            else:
                tmp = X[batch_sizes_cumsum[i-1]:batch_sizes_cumsum[i]].detach().cpu().numpy()
            X_full.append(tmp)
        pbmc.X = np.vstack(X_full)
        ari, nmi = tools.cluster_with_leiden(pbmc, resolution_values=[0.50, 0.60, 0.70, 0.80])
        print(f"Iteration {epoch + 1}/{args.epochs}: loss: {loss.item():.4f}, pearson: {pearson_corr:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}")
        wandb.log({"Iteration": epoch + 1, "loss": loss, "pearson": pearson_corr, "ari": ari, "nmi": nmi})
