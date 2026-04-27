#!/home/qyhu/anaconda3/envs/cu116torch112_1/bin/python
"""Reproduce ADM training with gradient tracking to debug shape issue."""
import sys, os, numpy as np
os.chdir("/home/qyhu/SPAGS")
sys.path.append('.')

import torch
import torch.nn.functional as F

# Load init data
init_data = np.load("data/spags_format/foot_50_3views/init_foot_50_3views.npy", allow_pickle=True)
print(f"Init data type: {type(init_data)}")
if isinstance(init_data, np.ndarray) and init_data.dtype == object:
    for i, item in enumerate(init_data):
        if isinstance(item, np.ndarray):
            print(f"  item[{i}]: shape={item.shape}, dtype={item.dtype}")
else:
    print(f"  shape={init_data.shape}, dtype={init_data.dtype}")

# Now try a simple gradient flow test
from r2_gaussian.gaussian.gaussian_model import GaussianModel
from r2_gaussian.utils.adm_module import create_adm_module

# Create model with a small number of points
N = 128
device = 'cuda:0'

model = GaussianModel(3).to(device)
# Manually set up the model parameters
model._density = torch.nn.Parameter(torch.randn(N, 1, device=device))
model._xyz = torch.nn.Parameter(torch.randn(N, 3, device=device) * 0.5)
model._scaling = torch.nn.Parameter(torch.ones(N, 3, device=device) * 0.1)
model._rotation = torch.nn.Parameter(torch.zeros(N, 4, device=device))
model._rotation.data[:, 0] = 1.0
model._features_dc = torch.nn.Parameter(torch.ones(N, 1, 3, device=device) * 0.5)

# First test: without ADM
d0 = model.get_density
print(f"\nWithout ADM: density shape={d0.shape}")

# Test: with ADM
adm = create_adm_module(grid_size=16, feat_dim=8, r_max=0.5, device=device)
model.adm_module = adm
model.adm_schedule_s = 1.0
model.adm_view_scale = 1.0

d1 = model.get_density
print(f"With ADM:    density shape={d1.shape}")

# Check if shapes are the same
print(f"\n_density shape: {model._density.shape}")
print(f"density_activation(_density) shape: {model.density_activation(model._density).shape}")

# Now test gradient flow
print("\n=== Testing gradient flow ===")
loss = d1.sum()
loss.backward()
print(f"Gradient flow: OK")
print(f"ADM module has grad: {adm.tri_plane.grid_xy.grad is not None}")
print(f"_density has grad: {model._density.grad is not None}")
print(f"Gradient shapes:")
print(f"  d(L)/d_density: {model._density.grad.shape}")
