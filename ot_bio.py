import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ot
import sys
# import wandb
import anndata
import scipy.stats as stats
from scipy.stats import pearsonr


epochs = 1000000
device = 'cuda:0'
### batch_size <= min(X1[0], X2[0])
batch_size = 6400

citeseq = anndata.read_h5ad("/workspace/ImputationOT/citeseq_processed-001.h5ad")

def zeromean(X, dim):
    # 创建一个掩码，标记出非零元素的位置
    nonzero_mask = X != 0
    
    # 用 0 替换张量中原来的零元素
    X_nonzero = X.clone()
    X_nonzero[~nonzero_mask] = 0
    
    # 计算非零元素的个数
    count_nonzero = nonzero_mask.sum(dim=dim, keepdim=True).float()
    
    # 计算非零元素的和
    sum_nonzero = X_nonzero.sum(dim=dim, keepdim=True)
    
    # 计算均值，忽略零值
    mean_nonzero = sum_nonzero / count_nonzero
    
    return mean_nonzero.squeeze(dim)

### TODO: 检查 X 的数据格式，是否是稀疏矩阵？
X = citeseq.X.toarray()
X = torch.tensor(X).to(device)
X = X[:41482]    # data in site1, site2
ground_truth = X.clone()

gex_indices = np.where(citeseq.var['feature_types'] == 'GEX')[0]
adt_indices = np.where(citeseq.var['feature_types'] == 'ADT')[0]
GEX = X[:41482, gex_indices] # AnnData object with n_obs × n_vars = 41482 × 13953
ADT = X[:41482, adt_indices] # AnnData object with n_obs × n_vars = 41482 × 134

site1_indices = np.where(citeseq.obs['Site'] == 'site1')[0]
site2_indices = np.where(citeseq.obs['Site'] == 'site2')[0]
X1 = X[site1_indices, :] # AnnData object with n_obs × n_vars = 16311 × 14087
X2 = X[site2_indices, :] # AnnData object with n_obs × n_vars = 25171 × 14087

### 只填补0数据 - 无法评估
# mask = (X[site1_indices, :][:, adt_indices] == 0).double()
# imps = (0.1 * torch.randn(mask.shape).double() + zeromean(X[site2_indices, :][:, adt_indices], 0).repeat(len(site1_indices), 1))[mask.bool()].float()

### 把已有数据也当作0数据进行全面填补
mask = torch.ones((41482, 14087), dtype=torch.bool).to(device)
mask[16311:, 13953:] = False       # mask X(1,1)
mask = ~mask

nonzero_mask = (X[16311:, 13953:] != 0).to(device)

# imps = (0.1 * torch.randn(X[site1_indices, :][:, adt_indices].shape).double() + zeromean(X[site2_indices, :][:, adt_indices], 0).repeat(len(site1_indices), 1)).float()
# imps.requires_grad = True
imps = torch.ones(mask.sum(), device=device, requires_grad=True)
optimizer = optim.Adam([imps])

print("start optimizing")
with open('results_bio.txt', 'w') as f:
    for epoch in range(epochs):
        X_imputed = X.detach().clone()
        X_imputed[mask] = imps

        indices1 = torch.randperm(16311 // 2, device=device)[:batch_size]
        X1 = X_imputed[:16311, :][indices1, :]
        X2 = X_imputed[16311:, :][indices1, :]

        indices2 = torch.randperm(13953, device=device)[:134]
        GEX = X_imputed[:, indices2]
        ADT = X_imputed[:, -134:]
        loss = 0.5 * ot.sliced_wasserstein_distance(X1, X2) + 0.5 * ot.sliced_wasserstein_distance(GEX, ADT)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        X_imputed = X.detach().clone()
        X_imputed[mask] = imps

        if (epoch + 1) % 100 == 0:
            pearson_corr = pearsonr(X_imputed[16311:, 13953:][nonzero_mask].detach().cpu().numpy(), ground_truth[16311:, 13953:][nonzero_mask].detach().cpu().numpy())[0]
            f.write(f"Iteration {epoch + 1}/{epochs}: loss: {loss.item():.4f}, pearson: {pearson_corr:4f}\n")
            f.flush()