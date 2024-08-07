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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from utils import tools

seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument("--use_wandb", default=False)
parser.add_argument("--dkm_iters", type=int, default=10)
parser.add_argument("--n_classes", type=int, default=9)
parser.add_argument("--dkm_eps", type=float, default=1e-4)
parser.add_argument("--dkm_temperature", type=float, default=1.0)
parser.add_argument("--aux_weight", type=float, default=0.01)
args = parser.parse_args()

epochs = 8000
device = 'cuda:0'
n_projections = 2000
batch_size = 5000
SITE1_CELL = 16311
SITE2_CELL = 25171
SITE3_CELL = 32029
SITE4_CELL = 16750
FILLED_GEX = 2000

if args.use_wandb:
    wandb.init(
        project="ot",
        name="c-4adt-ablation1",
        config={
            "dataset": "NIPS2021-Cite-seq",
            "epochs": epochs,
            "missing data": "site3 gex",
            "n_projections": 2000,
            "h_loss weight": args.aux_weight,
            "n_classes": args.n_classes,
            "comment": "delete swd between modalities"
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

optimizer = optim.Adam([imps], lr=0.1)
lambda_lr = lambda epoch: 1 if epoch < 1000 else 0.001 + (0.1 - 0.001) * (1 - (epoch - 1000) / (epochs - 1000))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
dkm = tools.DKM(num_clusters=args.n_classes, temperature=args.dkm_temperature, max_iters=args.dkm_iters, epsilon=args.dkm_eps).cuda()

h_loss = torch.zeros(1).to(device)

print("Start optimizing")
for epoch in range(epochs):
    X_imputed = X.detach().clone()
    X_imputed[mask] = imps

    if epoch == 0 and args.use_wandb:
        pearson_corr = pearsonr(X_imputed[-SITE4_CELL:, 2000:][nonzero_mask42].detach().cpu().numpy(), ground_truth[-SITE4_CELL:, 2000:][nonzero_mask42].detach().cpu().numpy())[0]
        citeseq.X = np.vstack((X_imputed[:SITE1_CELL + SITE2_CELL].detach().cpu().numpy(), X3, X_imputed[-SITE4_CELL:].detach().cpu().numpy()))
        ari, nmi = tools.clustering(citeseq)
        print(f"Initial pearson: {pearson_corr:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}")
        wandb.log({"Iteration": epoch, "loss": 0, "pearson": pearson_corr, "ari": ari, "nmi": nmi})

    X12 = X_imputed[:SITE1_CELL + SITE2_CELL]
    X4  = X_imputed[-SITE4_CELL:]
    # GEX = torch.transpose(X_imputed[:, :2000], 0, 1)
    # ADT = torch.transpose(X_imputed[:, 2000:], 0, 1)

    ### soft clustering assignment
    # indices = torch.randperm(SITE1_CELL + SITE2_CELL + SITE4_CELL, device=device)[:batch_size]
    # G_X1, G_X2, G_X3 = dkm(X_imputed[:, :2000][indices])
    # A_X1, A_X2, A_X3 = dkm(X_imputed[:, 2000:][indices])
    # _, predicted_labels = torch.max(G_X3, dim=1)
    # print(f"GEX, dkm num_clusters={args.n_classes}, temperature = {args.dkm_temperature}, max_iters = {args.dkm_iters}, epsilon={args.dkm_eps}")
    # print("clustering labels:", predicted_labels.cpu().numpy())
    # true_labels = citeseq.obs["cell_type"]
    # indices = indices.cpu().numpy()
    # unique_labels, encoded_labels = np.unique(true_labels.to_numpy()[indices], return_inverse=True)
    # print("true labels:", encoded_labels)
    # ari = adjusted_rand_score(predicted_labels.cpu().numpy(), encoded_labels)
    # nmi = normalized_mutual_info_score(predicted_labels.cpu().numpy(), encoded_labels)
    # print(ari, nmi)
    # sys.exit()
    ### hungarian matching loss
    # M = torch.cdist(torch.t(C1), torch.t(C2))
    # P = tools.gumbel_sinkhorn(M, tau=1, n_iter=5)
    # h_loss = (M * P).sum()
    # h_loss = F.cross_entropy(G_X3, A_X3)
    
    w_h = 0 if epoch <= 1000 else args.aux_weight
    # loss = (0.5 * ot.sliced_wasserstein_distance(X12, X4, n_projections=n_projections) +
    #         0.5 * ot.sliced_wasserstein_distance(GEX, ADT, n_projections=n_projections) +
    #         w_h * h_loss)
    loss = ot.sliced_wasserstein_distance(X12, X4, n_projections=n_projections)
    print(f"{epoch}: h_loss = {h_loss.item():.4f}, loss = {loss.item():.4f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 300 == 0 and args.use_wandb:
        X_imputed = X.detach().clone()
        X_imputed[mask] = imps
        
        pearson_corr = pearsonr(X_imputed[-SITE4_CELL:, 2000:][nonzero_mask42].detach().cpu().numpy(), ground_truth[-SITE4_CELL:, 2000:][nonzero_mask42].detach().cpu().numpy())[0]
        citeseq.X = np.vstack((X_imputed[:SITE1_CELL + SITE2_CELL].detach().cpu().numpy(), X3, X_imputed[-SITE4_CELL:].detach().cpu().numpy()))
        ari, nmi = tools.clustering(citeseq)
        print(f"Iteration {epoch + 1}/{epochs}: loss: {loss.item():.4f}, pearson: {pearson_corr:.4f}, ari: {ari:.4f}, nmi: {nmi:.4f}")
        wandb.log({"Iteration": epoch + 1, "loss": loss, "pearson": pearson_corr, "ari": ari, "nmi": nmi})
