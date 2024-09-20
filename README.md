### Enviroment
```
git config --global user.email "eliu11037@gmail.com"
git config --global user.name "Yuyu-Liu11037"
conda create -n ot
source activate ot
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install pot anndata scipy wandb scanpy igraph geomloss[full] argparse cvxpy
```

### Data
```
mkdir data
cd data
pip install gdown
gdown https://drive.google.com/uc?id=1raqlykXvm5wHjam1Up0SHYT-7gq7coz4
gdown https://drive.google.com/uc?id=1pilLsl2N1HX_US_Y6X6eAwmXaPso_1Mu
```

### Run experiment
```
# PBMCs
python imputationot/pbmc-3p/data_generation.py
python -m imputationot.pbmc-3p.pbmc --wandb_job ablation --wandb_name cells --use_wandb
python -m imputationot.pbmc-3p.pbmc --wandb_job aux --wandb_name equal_weighting --use_wandb --use_cluster --weights 0.5,0.5
```