import matplotlib.pyplot as plt
import torch
from torch import nn
import sys
import os
import shutil
import numpy as np

import Unet
import Plot
import Diffusion
import loader
import cfg
import joblib

# ====================================================================
# Config
# ====================================================================
DATASET = 'mnist_unet_diffusion'
config = cfg.load_config(DATASET)
# config.n_images = 2000
config.DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

suffix = '{:s}_{:d}_3class_newUnet/'.format(config.DATASET, config.n_images)

# Create save paths
path_images  = config.path_save + 'Images/'  + suffix
path_models  = config.path_save + 'Models/'  + suffix
path_history = config.path_save + 'History/' + suffix
os.makedirs(path_images,  exist_ok=True)
os.makedirs(path_models,  exist_ok=True)
os.makedirs(path_history, exist_ok=True)

# Back up scripts
for fname, dst in [
    ('run_Diffusion_MNIST.py', path_models + '_run_Diffusion.py'),
    ('loader.py',        path_models + '_loader.py'),
    ('cfg.py',           path_models + '_cfg.py'),
]:
    try:
        shutil.copy(fname, dst)
    except FileNotFoundError:
        print(f'Warning: could not back up {fname}')

# ====================================================================
# Data
# ====================================================================
trainset, testset = loader.load_MNIST(config, loadtest=True, include_list=config.dataset_params["classes"], props=config.dataset_params["props"])

# # Remap labels to 0, 1, ... for CrossEntropyLoss
# label_map = {orig: new for new, orig in enumerate(config.include_list)}
# trainset.targets = torch.tensor([label_map[t.item()] for t in trainset.targets])
# if testset is not None:
#     testset.targets = torch.tensor([label_map[t.item()] for t in testset.targets])

# ====================================================================
# Compute largest eigenvalue (Lambda)
# ====================================================================
full_loader = torch.utils.data.DataLoader(trainset,
                                          batch_size=len(trainset),
                                          shuffle=True,
                                          num_workers=1,
                                          pin_memory=False)
images, _ = next(iter(full_loader))
tt = images[:, 0, :, :].reshape(-1, np.prod(config.IMG_SHAPE[1:]))
cov = torch.cov(tt.T)
Lambda = torch.lobpcg(cov)[0].item()
print('Largest eigenvalue is {:.4f}'.format(Lambda))
joblib.dump({'config': config.dataset_params, 'Lambda': Lambda}, path_history + 'config.jbl', compress=3)

# ====================================================================
# Training loader
# ====================================================================
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=config.BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=1,
                                          pin_memory=False)

# ====================================================================
# Model
# ====================================================================
model = Unet.UNet(
    input_channels          = 1,  # mnist_unet_diffusion is grayscale
    output_channels         = 1,
    base_channels           = 64,
    base_channels_multiples = (1, 2, 4, 4),
    apply_attention         = (False, True, True, False),
    dropout_rate            = 0.1,
)
model.to(config.DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print('Model parameters: {:.2f}M'.format(n_params / 1e6))

# ====================================================================
# Diffusion training
# ====================================================================
optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
df = Diffusion.DiffusionConfig(
    n_steps   = config.TIMESTEPS,
    img_shape = config.IMG_SHAPE,
    device    = config.DEVICE,
)
loss_fn = nn.MSELoss()

save_every = 10000
Diffusion.train(model, trainloader, optimizer, config, df,
                loss_fn, save_every=save_every, suffix=suffix,
                data_snaps_steps=True)

