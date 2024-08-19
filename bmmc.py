import numpy as np
import torch
import torch.optim as optim
import anndata as ad
import scanpy as sc
import wandb
import sys
import random
import argparse
from scipy.stats import pearsonr
from geomloss import SamplesLoss

from utils import tools

parser = argparse.ArgumentParser()
parser.add_argument("--use_wandb", action="store_true", default=False)
parser.add_argument("--aux_weight", type=float, default=1)
parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--eval_interval", type=int, default=100)
parser.add_argument("--start_aux", type=int, default=400)
parser.add_argument("--batch_size", type=int, default=3000)
parser.add_argument("--seed", type=int, default=2024)
parser.add_argument("--dataset", type=str, choices=["citeseq", "multiome"], default="citeseq")
parser.add_argument("--missing_site", type=str, choices=["1", "2", "3", "4"])
parser.add_argument("--missing_omic", type=str, choices=["1", "2"], default="1")
parser.add_argument("--wandb_group", type=str, choices=["citeseq-3gex", "citeseq-4adt", "multiome-3atac", "multiome-4gex"], default="citeseq-4adt")
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
h_loss = torch.zeros(1).to(device)
omics_loss = torch.zeros(1).to(device)
cells_loss = torch.zeros(1).to(device)

if args.dataset == "citeseq":
    adata = ad.read_h5ad("/workspace/ImputationOT/data/citeseq_processed.h5ad")
    adata.var_names_make_unique()
    SITE1 = 16311
    SITE2 = 25171
    SITE3 = 32029
    SITE4 = 16750
    OMICS1 = 2000
    OMICS2 = 134
elif args.dataset == "multiome":
    adata = ad.read_h5ad("/workspace/ImputationOT/data/multiome_processed.h5ad")
    adata.var_names_make_unique()
    SITE1 = 17243
    SITE2 = 15226
    SITE3 = 14556
    SITE4 = 22224
    OMICS1 = 4000
    OMICS2 = 2832
else:
    adata = None

row_indices = [0, SITE1, SITE1 + SITE2, SITE1 + SITE2 + SITE3, SITE1 + SITE2 + SITE3 + SITE4]
col_indices = [0, OMICS1, OMICS1 + OMICS2]

if args.use_wandb is True:
    wandb.init(
        project="ot",
        group=args.wandb_group,
        job_type=args.wandb_job,
        name=args.wandb_name
    )

adata = tools.preprocess(adata)  # by default: [GEX ADT] or [ATAC GEX]
print(f"Finish preprocessing\n")

X = adata.X.toarray()
X = torch.tensor(X).to(device)
ground_truth = X.clone()

mask = torch.zeros(X.shape, dtype=torch.bool).to(device)
if args.missing_omic == "1":
    if args.missing_site == "1":
        mask[:SITE1, :OMICS1] = True
    elif args.missing_site == "2":
        mask[SITE1:SITE1 + SITE2, :OMICS1] = True
    elif args.missing_site == "3":
        mask[SITE1 + SITE2:SITE1 + SITE2 + SITE3, :OMICS1] = True
    elif args.missing_site == "4":
        mask[-SITE4:, :OMICS1] = True
elif args.missing_omic == "2":
    if args.missing_site == "1":
        mask[:SITE1, -OMICS2:] = True
    elif args.missing_site == "2":
        mask[SITE1:SITE1 + SITE2, -OMICS2:] = True
    elif args.missing_site == "3":
        mask[SITE1 + SITE2:SITE1 + SITE2 + SITE3, -OMICS2:] = True
    elif args.missing_site == "4":
        mask[-SITE4:, -OMICS2:] = True


def initialize_submatrix(X, s1, s2, s3, s4, o1, o2, target_row_part, target_col_part):
    # 确定行部分的起始和结束索引
    row_sizes = [s1, s2, s3, s4]
    row_start = sum(row_sizes[:target_row_part])
    row_end = row_start + row_sizes[target_row_part]

    # 确定列部分的起始和结束索引
    col_sizes = [o1, o2]
    col_start = sum(col_sizes[:target_col_part])
    col_end = col_start + col_sizes[target_col_part]

    # 创建一个掩码来排除被选中的行
    mask = torch.ones(X.shape[0], dtype=torch.bool)
    mask[row_start:row_end] = False

    # 初始化要返回的子矩阵
    submatrix = torch.zeros((row_end - row_start, col_end - col_start))

    # 对于每一列，计算其余行的非零值的平均值
    for col in range(col_start, col_end):
        non_zero_values = X[mask, col]
        non_zero_values = non_zero_values[non_zero_values != 0]
        if len(non_zero_values) > 0:
            mean_value = non_zero_values.mean()
        else:
            mean_value = 0
        submatrix[:, col - col_start] = mean_value

    return submatrix


imps = initialize_submatrix(X, SITE1, SITE2, SITE3, SITE4, OMICS1, OMICS2, int(args.missing_site) - 1,
                            int(args.missing_omic) - 1).to(device)
imps += torch.randn(imps.shape, device=device) * 0.1
imps.requires_grad = True


def lr_lambda(epoch, iter1=300, iter2=1000, init=0.1):
    if epoch < iter1:
        return init
    elif iter1 <= epoch < iter2:
        return init + 0.001 - (epoch - iter1) / (iter2 - iter1)
    else:
        return 0.001


optimizer = optim.Adam([imps], 1.0)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

print("Start optimizing")
for epoch in range(args.epochs):
    X_imputed = X.detach().clone()
    X_imputed[mask] = imps

    if epoch == 0 and args.use_wandb is True:
        prediction = X_imputed[row_indices[int(args.missing_site) - 1]:row_indices[int(args.missing_site)],
                     col_indices[int(args.missing_omic) - 1]:col_indices[int(args.missing_omic)]]
        gt = ground_truth[row_indices[int(args.missing_site) - 1]:row_indices[int(args.missing_site)],
             col_indices[int(args.missing_omic) - 1]:col_indices[int(args.missing_omic)]]
        pearson_corr = pearsonr(prediction[prediction != 0].detach().cpu().numpy(), gt[gt != 0].detach().cpu().numpy())[
            0]
        adata.X = X
        ari, nmi = tools.clustering(adata)
        print(f"Initial pearson: {pearson_corr:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}")
        wandb.log({"Iteration": epoch, "loss": 0, "pearson": pearson_corr, "ari": ari, "nmi": nmi})

    start = row_indices[int(args.missing_site) - 1]
    end = row_indices[int(args.missing_site)]
    rows_in_part = X[start:end, :]
    data_cells1 = rows_in_part[torch.randperm(rows_in_part.size(0))[::args.batch_size], :]
    rows_outside_part = torch.cat([X[:start, :], X[end:, :]], dim=0)
    data_cells2 = rows_outside_part[torch.randperm(rows_outside_part.size(0))[::args.batch_size], :]
    data_omic1 = torch.transpose(X_imputed[:, :OMICS1], 0, 1)
    data_omic2 = torch.transpose(X_imputed[:, OMICS1:], 0, 1)

    if epoch >= args.start_aux:
        if epoch % args.eval_interval == 0:
            ### calculate cluster results
            labels1 = tools.calculate_cluster_labels(data_cells1)
            labels2 = tools.calculate_cluster_labels(data_cells2)
        ### calculate cluster centroids
        centroids1 = tools.calculate_cluster_centroids(data_cells1, labels1)
        centroids2 = tools.calculate_cluster_centroids(data_cells2, labels2)
        ### calculate cluster loss
        M = torch.cdist(centroids1, centroids2)
        P = tools.gumbel_sinkhorn(M, tau=1, n_iter=5)
        h_loss = (M * P).sum()

    w_h = 0 if epoch < args.start_aux else args.aux_weight
    omics_loss = SamplesLoss()(data_omic1, data_omic2)
    cells_loss = SamplesLoss()(data_cells1, data_cells2)
    loss = 0.5 * 0.001 * omics_loss + 0.5 * cells_loss + w_h * h_loss
    print(
        f"{epoch}: omics_loss = {omics_loss.item():.4f}, cells_loss = {cells_loss.item():.4f}, h_loss = {h_loss.item():.4f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % args.eval_interval == 0 and args.use_wandb is True:
        X_imputed = X.detach().clone()
        X_imputed[mask] = imps

        prediction = X_imputed[row_indices[int(args.missing_site) - 1]:row_indices[int(args.missing_site)],
                     col_indices[int(args.missing_omic) - 1]:col_indices[int(args.missing_omic)]]
        gt = ground_truth[row_indices[int(args.missing_site) - 1]:row_indices[int(args.missing_site)],
             col_indices[int(args.missing_omic) - 1]:col_indices[int(args.missing_omic)]]
        pearson_corr = pearsonr(prediction[prediction != 0].detach().cpu().numpy(), gt[gt != 0].detach().cpu().numpy())[
            0]
        adata.X = X
        ari, nmi = tools.clustering(adata)
        print(
            f"Iteration {epoch + 1}/{args.epochs}: loss: {loss.item():.4f}, pearson: {pearson_corr:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}")
        wandb.log({"Iteration": epoch + 1, "loss": loss, "pearson": pearson_corr, "ari": ari, "nmi": nmi})
