import torch
import anndata
import numpy as np
import ot
import random
import geomloss


from geomloss import SamplesLoss

# Hyperparameters setting

n_iters = 100
eps = 1e-5
scaling = 1

citeseq = anndata.read_h5ad("/workspace/citeseq_processed-001.h5ad")

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

X = citeseq.X.toarray()
X = torch.tensor(X)
ground_truth = X.clone()

gex_indices = np.where(citeseq.var['feature_types'] == 'GEX')[0]
adt_indices = np.where(citeseq.var['feature_types'] == 'ADT')[0]
GEX = X[:, gex_indices] # AnnData object with n_obs × n_vars = 69249 × 13953
ADT = X[:, adt_indices] # AnnData object with n_obs × n_vars = 69249 × 134

site1_indices = np.where(citeseq.obs['Site'] == 'site1')[0]
site2_indices = np.where(citeseq.obs['Site'] == 'site2')[0]
X1 = X[site1_indices, :] # AnnData object with n_obs × n_vars = 16311 × 14087
X2 = X[site2_indices, :] # AnnData object with n_obs × n_vars = 25171 × 14087

# 只填补0数据 - 无法评估
# mask = (X[site1_indices, :][:, adt_indices] == 0).double()
# imps = (0.1 * torch.randn(mask.shape).double() + zeromean(X[site2_indices, :][:, adt_indices], 0).repeat(len(site1_indices), 1))[mask.bool()].float()

# 把已有数据也当作0数据进行全面填补
imps = (0.1 * torch.randn(X[site1_indices, :][:, adt_indices].shape).double() + zeromean(X[site2_indices, :][:, adt_indices], 0).repeat(len(site1_indices), 1)).float()
imps.requires_grad = True

# Impute data & calculate loss
# X: inpute gene-cell matrix

optimizer = torch.optim.Adam([imps])
loss = 0
        
# Method: distance between D1, D2 + distance between GEX, ADT

def pearson(A, B):
    # Step 1: 计算均值
    mean_A = torch.mean(A, dim=0)
    mean_B = torch.mean(B, dim=0)
    
    # Step 2: 中心化数据
    A_centered = A - mean_A
    B_centered = B - mean_B
    
    # Step 3: 计算协方差
    covariance = torch.sum(A_centered * B_centered, dim=0)
    
    # Step 4: 计算标准差
    std_A = torch.sqrt(torch.sum(A_centered ** 2, dim=0))
    std_B = torch.sqrt(torch.sum(B_centered ** 2, dim=0))
    
    # Step 5: 计算 Pearson 相似分数
    pearson_score = covariance / (std_A * std_B)
    
    return pearson_score.mean()

print("start optimizing")
for i in range(n_iters):
    loss = 0
    X_imputed = X.detach().clone()
    assert X.dtype == imps.dtype, "dtype of X and imps should match"
    X_imputed[:16311, 13953:] = imps

    # print(MAE(imps, ground_truth[:16311, 13953:], mask).data)

    X1 = X_imputed[16211:16311, 13752:]
    X2 = X_imputed[16311:16411, 13752:]
    # GEX = X_imputed[:, :1]
    # ADT = X_imputed[:, -1:]
    assert X1.shape == X2.shape, "shape of X1 and X2 do not match"
    # assert GEX.shape == ADT.shape, "shape of GEX and ADT do not match"

    loss += SamplesLoss("sinkhorn", p=2)(X1, X2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Iteration {i + 1}/{n_iters}: loss = {loss.item()}")
    print(pearson(imps, ground_truth[:16311, 13953:]).item())