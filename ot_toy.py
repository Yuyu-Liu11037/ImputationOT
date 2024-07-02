import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ot
import sys
import wandb
import scipy.stats as stats
from scipy.stats import pearsonr

# wandb.init(project="ot11037",
#            name="ot_toy",)

m, n = 10000, 8000  # Matrix dimensions
k = 300
mask_size = (10000, 1)

device = 'cuda:0'

matrix = torch.randn(m, n).to(device)

original_matrix = matrix.clone()

mask = torch.ones((m, n), dtype=torch.bool).to(device)
mask[:mask_size[0], :mask_size[1]] = False
mask = ~mask

imps = torch.ones(mask.sum(), device=device, requires_grad=True)
optimizer = optim.Adam([imps], lr=0.01)

def MAE(X, X_true, mask):
    if torch.is_tensor(mask):
        mask_ = mask.bool()
        return torch.abs(X[mask_] - X_true[mask_]).sum() / mask_.sum()
    else: # should be an ndarray
        mask_ = mask.astype(bool)
        return np.absolute(X[mask_] - X_true[mask_]).sum() / mask_.sum()

num_epochs = 100000
with open('results.txt', 'w') as f:
    for epoch in range(num_epochs):
        X_filled = matrix.detach().clone()
        X_filled[mask] = imps
        if epoch == 0:
            pearson_corr_initial = pearsonr(imps.detach().cpu().numpy(), original_matrix[mask].detach().cpu().numpy())[0]
            f.write(f"Initial Pearson Correlation: {pearson_corr_initial:.4f}\n")
            anderson_test = stats.anderson(imps.detach().cpu().numpy().flatten(), dist='norm')
            f.write(f"Anderson-Darling Test: Statistic={anderson_test.statistic}, Critical Values={anderson_test.critical_values}\n")
        
        indices1 = torch.randperm(m // 2, device=device)[:1]
        
        X1 = X_filled[:m // 2, :][indices1, :]
        X2 = X_filled[m // 2:, :][indices1, :]
        
        loss = ot.sliced_wasserstein_distance(X1, X2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            pearson_corr = pearsonr(imps.detach().cpu().numpy(), original_matrix[mask].detach().cpu().numpy())[0]
            mae = MAE(X_filled, original_matrix, mask).mean().item()
            f.write(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Pearson Correlation: {pearson_corr:.4f}, MAE: {mae:.4f}\n")
        if (epoch + 1) % 1000 == 0:
            anderson_test = stats.anderson(imps.detach().cpu().numpy().flatten(), dist='norm')
            f.write(f"Anderson-Darling Test: Statistic={anderson_test.statistic}, Critical Values={anderson_test.critical_values}\n")
            # wandb.log({"loss": loss, "pearsonr": pearson_corr, "mae": mae})

        X_filled = matrix.detach().clone()
        X_filled[mask] = imps

# Save X_filled and original_matrix to npy files
np.save('X_filled.npy', X_filled.detach().cpu().numpy())
np.save('original_matrix.npy', original_matrix.detach().cpu().numpy())
