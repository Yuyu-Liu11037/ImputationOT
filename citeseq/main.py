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

def str_to_float_list(arg):
    return [float(x) for x in arg.split(',')]

parser = argparse.ArgumentParser()
parser.add_argument("--use_wandb", action="store_true", default=False)
parser.add_argument("--aux_weight", type=float, default=0)
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--eval_interval", type=int, default=5)
parser.add_argument("--start_aux", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=3000)
parser.add_argument("--seed", type=int, default=2024)
parser.add_argument("--resolution_values", type=str_to_float_list, default=[0.05, 0.1, 2, 5, 10])

parser.add_argument("--source_batches", type=str, default="1,2", help="Impute batch range from 1 to 7")
parser.add_argument("--target_batch", type=int, default=3, help="Batch number to impute")

parser.add_argument("--wandb_group", type=str, default="citeseq")
parser.add_argument("--wandb_job", type=str, choices=["main", "ablation", "aux"], default="main")
parser.add_argument("--wandb_name", type=str, default="1,2-3")
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
batch_sizes = [16311, 25171, 32029, 16750]
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
            "dataset": "citeseq1",
            "epochs": args.epochs,
            "h_loss weight": args.aux_weight
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
X = torch.tensor(X).to(device)
X_source = torch.cat([X[batch_sizes_cumsum[i - 1]:batch_sizes_cumsum[i]] for i in source_batches], dim=0)
X_target = X[batch_sizes_cumsum[target_batch - 1]:batch_sizes_cumsum[target_batch]]
ground_truth = X_target.clone()

mask = torch.zeros(X_target.shape, dtype=torch.bool).to(device)
mask[:, :FILLED_GEX] = True
nonzero_mask_target = (X_target[:, :FILLED_GEX] != 0).to(device) 
nonzero_mask1121 = (X[:SITE1_CELL + SITE2_CELL, :FILLED_GEX] != 0).to(device)   # nonzero data of X(1,1), X(2,1)
nonzero_mask31 = (X[-SITE3_CELL:, :FILLED_GEX] != 0).to(device)   # nonzero data of X(3,1)
nonzero_mask32 = (X[-SITE3_CELL:, FILLED_GEX:] != 0).to(device)   # nonzero data of X(3,2)

mean_values = torch.sum(X[:SITE1_CELL + SITE2_CELL, :FILLED_GEX], dim=0) / torch.sum(nonzero_mask1121, dim=0)
imps = mean_values.repeat(SITE3_CELL).to(device)
imps += torch.randn(imps.shape, device=device) * 0.1
imps.requires_grad = True

### best performance
def lr_lambda(epoch):
    if epoch < 10:
        return 0.1
    elif 10 <= epoch < 50:
        return 0.101 - (epoch - 10) / 400.0
    else:
        return 0.001

optimizer = optim.Adam([imps], 1.0)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

h_loss = torch.zeros(1).to(device)
cells_loss = torch.zeros(1).to(device)

print(f"Start optimizing: use batch(es) {args.source_batches} to impute batch {args.target_batch}")
for epoch in range(args.epochs):
    X_imputed = X_target.detach().clone()
    X_imputed[mask] = imps

    # if epoch == 0 and args.use_wandb is True:
    if epoch == 0:
        pearson_corr = pearsonr(X_imputed[:, :FILLED_GEX][nonzero_mask_target].detach().cpu().numpy(), ground_truth[:, :FILLED_GEX][nonzero_mask_target].detach().cpu().numpy())[0]
        mae, rmse = tools.calculate_mae_rmse(X_imputed[:, :FILLED_GEX], ground_truth[:, :FILLED_GEX], nonzero_mask_target)
        X_full = []
        for i in range(1, len(batch_sizes) + 1):
            if i in source_batches:
                tmp = X[batch_sizes_cumsum[i-1]:batch_sizes_cumsum[i]].detach().cpu().numpy()
            elif i == target_batch:
                tmp = X_imputed.detach().cpu().numpy()
            else:
                tmp = X[batch_sizes_cumsum[i-1]:batch_sizes_cumsum[i]].detach().cpu().numpy()
            X_full.append(tmp)
        citeseq.X = np.vstack(X_full)
        ari, nmi, purity, jaccard = tools.cluster_with_leiden(citeseq, resolution_values=args.resolution_values)
        print(f"Initial pearson: {pearson_corr:.4f}, mae: {mae:.4f}, rmse: {rmse:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}, purity: {purity:.4f}, jaccard: {jaccard:.4f}")
        wandb.log({"Iteration": 0, "loss": 0, "pearson": pearson_corr, "mae": mae, "rmse": rmse, "ari": ari, "nmi": nmi, "purity": purity, "jaccard": jaccard})

    indices1 = torch.randperm(X_source.shape[0], device=device)[:args.batch_size]
    indices2 = torch.randperm(X_imputed.shape[0], device=device)[:args.batch_size]
    X1 = X_source[indices1]
    X2 = X_imputed[indices2]
    
    w_h = 0 if epoch < args.start_aux else args.aux_weight
    cells_loss = SamplesLoss()(X1, X2)
    loss = cells_loss + w_h * h_loss
    lr = lr_lambda(epoch)
    print(f"{epoch}: lr = {lr:.4f}, cells_loss = {cells_loss.item():.4f}, h_loss = {h_loss.item():.4f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    # if (epoch + 1) % args.eval_interval == 0 and args.use_wandb is True:
    #     X_imputed = imps
        
    #     pearson_corr = pearsonr(X_imputed[nonzero_mask_target].detach().cpu().numpy(), ground_truth[nonzero_mask_target].detach().cpu().numpy())[0]
    #     mae, rmse = tools.calculate_mae_rmse(X_imputed, ground_truth, nonzero_mask_target)
    #     X_full = []
    #     for i in range(1, len(batch_sizes) + 1):
    #         if i in source_batches:
    #             tmp = X[batch_sizes_cumsum[i-1]:batch_sizes_cumsum[i]].detach().cpu().numpy()
    #         elif i == target_batch:
    #             tmp = X_imputed.detach().cpu().numpy()
    #         else:
    #             tmp = X[batch_sizes_cumsum[i-1]:batch_sizes_cumsum[i]].detach().cpu().numpy()
    #         X_full.append(tmp)
    #     citeseq.X = np.vstack(X_full)
    #     ari, nmi, purity, jaccard = tools.cluster_with_leiden(citeseq, resolution_values=args.resolution_values)
    #     print(f"Iteration {epoch + 1}/{args.epochs}: loss: {loss.item():.4f}, pearson: {pearson_corr:.4f}, mae: {mae:.4f}, rmse: {rmse:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}, purity: {purity:.4f}, jaccard: {jaccard:.4f}")
    #     wandb.log({"Iteration": epoch + 1, "loss": loss, "pearson": pearson_corr, "mae": mae, "rmse": rmse, "ari": ari, "nmi": nmi, "purity": purity, "jaccard": jaccard})
