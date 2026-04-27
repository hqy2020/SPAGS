#!/home/qyhu/anaconda3/envs/cu116torch112_1/bin/python
"""Debug the actual init pipeline shapes."""
import sys, os, numpy as np
os.chdir("/home/qyhu/SPAGS")
sys.path.append('.')

import torch
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")

# Load init point cloud
init_data = np.load("data/spags_format/foot_50_3views/init_foot_50_3views.npy", allow_pickle=True)
print(f"\nInit data type: {type(init_data)}")
if isinstance(init_data, np.ndarray) and init_data.dtype == object:
    print(f"Object array shape: {init_data.shape}")
    for i, item in enumerate(init_data):
        if isinstance(item, np.ndarray):
            print(f"  item[{i}]: shape={item.shape}, dtype={item.dtype}")
        else:
            print(f"  item[{i}]: type={type(item)}")
elif isinstance(init_data, np.ndarray):
    print(f"ndarray shape: {init_data.shape}, dtype={init_data.dtype}")
else:
    print(f"Type: {type(init_data)}")

# Try loading the meta_data to understand
import json
with open("data/spags_format/foot_50_3views/meta_data.json") as f:
    meta = json.load(f)
print(f"\nMeta keys: {list(meta.keys())}")
print(f"nVoxel: {meta.get('nVoxel')}")
