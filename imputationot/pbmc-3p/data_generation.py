import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
import torch.nn as nn
import torch.optim as optim
import ot
import sys
import os
import tarfile
import scipy.io
import h5py
import gzip
from scipy.sparse import issparse
from scipy.sparse import csr_matrix


barcodes_file = '/workspace/ImputationOT/imputationot/data/GSM5008737_RNA_3P-barcodes.tsv.gz'
features_file = '/workspace/ImputationOT/imputationot/data/GSM5008737_RNA_3P-features.tsv.gz'
matrix_file = '/workspace/ImputationOT/imputationot/data/GSM5008737_RNA_3P-matrix.mtx.gz'

barcodes = pd.read_csv(barcodes_file, header=None, sep='\t')
features = pd.read_csv(features_file, header=None, sep='\t')
matrix = scipy.io.mmread(matrix_file)

adata_rna = ad.AnnData(X=matrix.T)

adata_rna.obs['barcodes'] = barcodes[0].values
adata_rna.var['symbols'] = features[0].values

data_3P = pd.read_csv('/workspace/ImputationOT/imputationot/data/meta_data_3P.csv.gz')
data_3P.set_index('Unnamed: 0', inplace=True)
adata_rna.obs = adata_rna.obs.merge(data_3P, left_on='barcodes', right_index=True, how='left')
adata_rna.var['feature_type'] = 'RNA'

print(adata_rna)

barcodes_file = '/workspace/ImputationOT/imputationot/data/GSM5008738_ADT_3P-barcodes.tsv.gz'
features_file = '/workspace/ImputationOT/imputationot/data/GSM5008738_ADT_3P-features.tsv.gz'
matrix_file = '/workspace/ImputationOT/imputationot/data/GSM5008738_ADT_3P-matrix.mtx.gz'

barcodes = pd.read_csv(barcodes_file, header=None, sep='\t')
features = pd.read_csv(features_file, header=None, sep='\t')
matrix = scipy.io.mmread(matrix_file)

adata_adt = ad.AnnData(X=matrix.T)

adata_adt.obs['barcodes'] = barcodes[0].values
adata_adt.var['symbols'] = features[0].values

data_3P = pd.read_csv('/workspace/ImputationOT/imputationot/data/meta_data_3P.csv.gz')
data_3P.set_index('Unnamed: 0', inplace=True)
adata_adt.obs = adata_adt.obs.merge(data_3P, left_on='barcodes', right_index=True, how='left')
adata_adt.var['feature_type'] = 'ADT'
print(adata_adt)

adata_combined = ad.concat([adata_rna, adata_adt], axis=1, merge='same')
adata_combined.write("/workspace/ImputationOT/imputationot/data/pbmc2.h5ad")
print(adata_combined)