import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ot
import anndata as ad
import scanpy as sc
import wandb
import random
import argparse
from scipy.stats import pearsonr
from geomloss import SamplesLoss

def str_to_float_list(arg):
    return [float(x) for x in arg.split(',')]

from imputationot.utils import correlation_matrix, correlation_matrix_distance, calculate_mae_rmse, calculate_cluster_labels, calculate_cluster_centroids, cluster_with_leiden
from imputationot.weighting import RLW

parser = argparse.ArgumentParser()
parser.add_argument("--use_wandb", action="store_true", default=False)
parser.add_argument("--use_cluster", action="store_true", default=False)
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--eval_interval", type=int, default=5)
parser.add_argument("--weights", type=str_to_float_list, default=[0.5,0.5])
parser.add_argument("--batch_size", type=int, default=3000)
parser.add_argument("--seed", type=int, default=2024)
parser.add_argument("--resolution_values", type=str_to_float_list, default=[0.7, 0.8, 0.9, 1, 1.1])

parser.add_argument("--source_batches", type=str, default="1", help="Impute batch size range from 1 to 9")
parser.add_argument("--target_batch", type=int, default=2, help="Batch number to impute")

parser.add_argument("--wandb_group", type=str, default="pbmc-partial")
parser.add_argument("--wandb_job", type=str, choices=["main", "ablation", "aux"], default="aux")
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
        name=args.wandb_name
    )

pbmc = ad.read_h5ad("/workspace/ImputationOT/imputationot/data/vento_pbmc_processed.h5ad") 
pbmc.var_names_make_unique()
missing_num = pbmc.X.shape[1] // 2

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

ground_truth = X_target.clone()

mask = torch.zeros(X_target.shape, dtype=torch.bool).to(device)
mask[:, :missing_num] = True
nonzero_mask_source = (X_source[:, :missing_num] != 0).to(device)
nonzero_mask_target = (X_target[:, :missing_num] != 0).to(device) 

epsilon = 1e-8
mean_values = torch.sum(X_source[:, :missing_num] * nonzero_mask_source, dim=0) / (torch.sum(nonzero_mask_source, dim=0) + epsilon)
imps = mean_values.repeat(X_target.shape[0]).to(device)
imps += torch.randn(imps.shape, device=device) * 0.1
imps.requires_grad = True
# imps = imps.view(X_target.shape).detach().requires_grad_()

### best performance
def lr_lambda(epoch):
    if epoch < 10:
        return 0.1
    elif 10 <= epoch < 100:
        return 0.101 - (epoch - 10) / 900.0
    else:
        return 0.001

optimizer = optim.Adam([imps], 1.0)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
grad_fn = RLW()
grad_fn.init_param()

h_loss = torch.zeros(1).to(device)
cells_loss = torch.zeros(1).to(device)

print(f"Start optimizing: use batch(es) {args.source_batches} to impute batch {args.target_batch}")
for epoch in range(args.epochs):
    optimizer.zero_grad()
    X_imputed = X_target.detach().clone()
    X_imputed[mask] = imps

    if epoch == 0 and args.use_wandb is True:
        pearson_corr = pearsonr(X_imputed[nonzero_mask_target].detach().cpu().numpy(), ground_truth[nonzero_mask_target].detach().cpu().numpy())[0]
        mae, rmse = calculate_mae_rmse(X_imputed[:, :missing_num], ground_truth[:, :missing_num], nonzero_mask_target)
        X_full = []
        for i in range(1, len(batch_sizes) + 1):
            if i in source_batches:
                tmp = X[batch_sizes_cumsum[i-1]:batch_sizes_cumsum[i]].detach().cpu().numpy()
            elif i == target_batch:
                tmp = X_imputed.detach().cpu().numpy()
            else:
                tmp = X[batch_sizes_cumsum[i-1]:batch_sizes_cumsum[i]].detach().cpu().numpy()
            X_full.append(tmp)
        pbmc.X = np.vstack(X_full)
        ari, nmi, purity, jaccard = cluster_with_leiden(pbmc, resolution_values=args.resolution_values, tag="Annotation")
        print(f"Initial pearson: {pearson_corr:.4f}, mae: {mae:.4f}, rmse: {rmse:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}, purity: {purity:.4f}, jaccard: {jaccard:.4f}")
        wandb.log({"Iteration": 0, "loss": 0, "pearson": pearson_corr, "mae": mae, "rmse": rmse, "ari": ari, "nmi": nmi, "purity": purity, "jaccard": jaccard})

    indices1 = torch.randperm(X_source.shape[0], device=device)[:args.batch_size]
    indices2 = torch.randperm(X_target.shape[0], device=device)[:args.batch_size]
    X1 = X_source[indices1]
    X2 = X_imputed[indices2]
    cells_loss = SamplesLoss()(X1, X2)

    # if args.use_cluster is True:
    #     ### calculate cluster results
    #     labels1 = calculate_cluster_labels(X1, resolution=0.9)
    #     labels2 = calculate_cluster_labels(X2, resolution=0.9)
    #     ### calculate cluster centroids
    #     centroids1 = calculate_cluster_centroids(X1, labels1)
    #     centroids2 = calculate_cluster_centroids(X2, labels2)
    #     ### calculate cluster loss
    #     h_loss = SamplesLoss()(centroids1, centroids2)
    
    # loss = args.weights[0] * cells_loss + args.weights[1] * h_loss
    # losses = torch.stack([cells_loss, h_loss])
    # sol = grad_fn.backward(losses)
    # print(sol)
    # loss = sol[0] * cells_loss + sol[1] * h_loss
    loss = cells_loss
    lr = lr_lambda(epoch)
    print(f"{epoch}: lr = {lr:.4f}, cells_loss = {cells_loss.item():.4f}, h_loss = {h_loss.item():.4f}")

    loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % args.eval_interval == 0 and args.use_wandb is True:
        X_imputed[mask] = imps
        
        pearson_corr = pearsonr(X_imputed[nonzero_mask_target].detach().cpu().numpy(), ground_truth[nonzero_mask_target].detach().cpu().numpy())[0]
        mae, rmse = calculate_mae_rmse(X_imputed[:, :missing_num], ground_truth[:, :missing_num], nonzero_mask_target)
        X_full = []
        for i in range(1, len(batch_sizes) + 1):
            if i in source_batches:
                tmp = X[batch_sizes_cumsum[i-1]:batch_sizes_cumsum[i]].detach().cpu().numpy()
            elif i == target_batch:
                tmp = X_imputed.detach().cpu().numpy()
            else:
                tmp = X[batch_sizes_cumsum[i-1]:batch_sizes_cumsum[i]].detach().cpu().numpy()
            X_full.append(tmp)
        pbmc.X = np.vstack(X_full)
        ari, nmi, purity, jaccard = cluster_with_leiden(pbmc, resolution_values=args.resolution_values, tag="Annotation")
        print(f"Iteration {epoch + 1}/{args.epochs}: loss: {loss.item():.4f}, pearson: {pearson_corr:.4f}, mae: {mae:.4f}, rmse: {rmse:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}, purity: {purity:.4f}, jaccard: {jaccard:.4f}")
        wandb.log({"Iteration": epoch + 1, "loss": loss, "pearson": pearson_corr, "mae": mae, "rmse": rmse, "ari": ari, "nmi": nmi, "purity": purity, "jaccard": jaccard})
