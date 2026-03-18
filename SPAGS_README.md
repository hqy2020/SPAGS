&nbsp;

<div align="center">

<h2> SPAGS: Sparse-view Prior-Augmented Gaussian Splatting for CT Reconstruction </h2>

*A unified framework integrating innovations from R²-Gaussian, X-Gaussian, DNGaussian, CoR-GS, and FSGS for sparse-view medical CT 3D reconstruction.*

</div>

&nbsp;

## Introduction

SPAGS (Sparse-view Prior-Augmented Gaussian Splatting) is a research framework for **sparse-view medical CT reconstruction** using 3D Gaussian Splatting. It integrates key innovations from five state-of-the-art methods into a unified, modular architecture built on the [R²-Gaussian](https://github.com/Ruyi-Zha/r2_gaussian) (NeurIPS 2024) baseline.

### Integrated Methods

| Method | Paper | Venue | Key Innovation Integrated |
|--------|-------|-------|--------------------------|
| **R²-Gaussian** | [arXiv](https://arxiv.org/abs/2405.20693) | NeurIPS 2024 | Radiative Gaussian Splatting baseline for CT |
| **X-Gaussian** | [arXiv](https://arxiv.org/abs/2403.04116) | ECCV 2024 | X-ray differentiable rendering pipeline |
| **DNGaussian** | [arXiv](https://arxiv.org/abs/2403.06051) | CVPR 2024 | Global-Local Depth Normalization, Dual-Phase Depth |
| **CoR-GS** | [arXiv](https://arxiv.org/abs/2405.12163) | ECCV 2024 | Multi-model Co-Regularization, Co-Pruning |
| **FSGS** | [arXiv](https://arxiv.org/abs/2312.00451) | CVPR 2024 | Proximity-guided Densification, Pseudo-view Depth |

### SPAGS Innovation Axes

SPAGS explores three innovation axes for sparse-view CT:

1. **ADM (Adaptive Densification Module)**: View-aware densification combining FSGS proximity-guided unpooling with R²-Gaussian gradient-based densification.
2. **SPS (Sparse-view Prior Synthesis)**: CoR-GS co-regularization + FSGS pseudo-view generation for novel view supervision.
3. **GAR (Geometry-Aware Regularization)**: DNGaussian global-local depth normalization + Gaussian shape regularization + graph Laplacian smoothness.

## Architecture

```
SPAGS Framework
├── R²-Gaussian Baseline          # Radiative 3DGS for CT (train.py)
│   ├── X-ray Volume Rendering     # Density-based (not opacity-based) splatting
│   ├── Multi-Gaussian Training    # N independent Gaussian fields
│   └── TV Regularization          # Total variation on density volume
│
├── ADM: Adaptive Densification    # (r2_gaussian/utils/fsgs_proximity*.py)
│   ├── Gradient-based Clone/Split # Standard 3DGS densification
│   ├── Proximity-guided Unpooling # FSGS: fill spatial gaps (new Gaussians at midpoints)
│   └── Distance-based Splitting   # FSGS: split isolated large Gaussians
│
├── SPS: Sparse-view Prior         # (r2_gaussian/utils/pseudo_view_utils.py, fsgs_complete.py)
│   ├── SLERP View Interpolation   # FSGS: smooth camera path interpolation
│   ├── Co-Regularization Loss     # CoR-GS: cross-model photometric consistency
│   ├── Pseudo-view Generation     # Novel view synthesis for data augmentation
│   └── Co-Pruning                 # CoR-GS: geometric consistency pruning
│
├── GAR: Geometry Regularization   # (r2_gaussian/utils/dngaussian_utils.py, loss_utils.py)
│   ├── Global-Local Depth Loss    # DNGaussian: Pearson + patch-wise MSE
│   ├── Dual-Phase Depth           # DNGaussian: hard→soft depth schedule
│   ├── Gaussian Shape Reg.        # DNGaussian: prevent degenerate aspect ratios
│   ├── Graph Laplacian            # CoR-GS: density smoothness on KNN graph
│   └── Depth Correlation Loss     # X-Gaussian: Pearson on rendered vs. GT depth
│
└── Evaluation                     # (test.py)
    ├── 2D Metrics                 # PSNR, SSIM on X-ray projections
    └── 3D Metrics                 # PSNR, SSIM on CT volume slices
```

## Installation

```bash
# Clone repository
git clone https://github.com/hqy2020/SPAGS.git --recursive
cd SPAGS

# Create conda environment
conda create -n spags python=3.9 -y
conda activate spags

# Install PyTorch (adjust for your CUDA version)
# For RTX 4090 / CUDA 12.4:
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
# For RTX 3090 / CUDA 11.8:
# pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install opencv-python matplotlib pydicom SimpleITK \
    open3d==0.18.0 plyfile tensorboard tensorboardX \
    pyyaml tqdm scikit-image numpy==1.24.1

# Compile CUDA submodules
cd r2_gaussian/submodules/simple-knn && python setup.py install && cd ../../..
cd r2_gaussian/submodules/xray-gaussian-rasterization-voxelization && pip install -e . && cd ../../..

# (Optional) Install TIGRE for data generation
pip install tigre==2.3
```

## Data

Download datasets from [Google Drive](https://drive.google.com/drive/folders/1W46wpeN7byWLC0f3cGIvoT_xbwT1b7gZ):
- Format: `{organ}_50_{N}views.pickle`
- Organs: chest, foot, head, abdomen, pancreas
- Views: 3, 6, 9

## Usage

### Basic Training (R²-Gaussian baseline)

```bash
python train.py -s data/foot_50_3views.pickle -m output/foot_3v_baseline --gaussiansN 1
```

### With SPAGS Modules

```bash
# ADM: Proximity-guided densification (FSGS)
python train.py -s data/foot_50_3views.pickle -m output/foot_3v_adm \
    --gaussiansN 1 --enable_fsgs_proximity_guided --enable_fsgs_complete_system

# SPS: Co-regularization (CoR-GS style, 2 Gaussian models)
python train.py -s data/foot_50_3views.pickle -m output/foot_3v_sps \
    --gaussiansN 2 --multi_gaussian_weight 0.1

# GAR: Depth regularization (DNGaussian style)
python train.py -s data/foot_50_3views.pickle -m output/foot_3v_gar \
    --gaussiansN 1 --enable_depth --depth_loss_weight 0.1 --depth_loss_type pearson

# Full SPAGS (all modules)
python train.py -s data/foot_50_3views.pickle -m output/foot_3v_spags \
    --gaussiansN 2 --multi_gaussian_weight 0.1 \
    --enable_depth --depth_loss_weight 0.1 \
    --enable_fsgs_pseudo_labels --enable_fsgs_proximity_guided --enable_fsgs_complete_system
```

### Evaluation

```bash
python test.py -m output/foot_3v_spags
# Results in: output/foot_3v_spags/eval/iter_030000/eval2d_render_test.yml
```

### Batch Experiments (5 methods x 5 organs x 3 views = 75 experiments)

```bash
bash scripts/batch_run_all_experiments.sh --skip-existing
# Or filter:
bash scripts/batch_run_all_experiments.sh --method R2GS --organ foot --views 3
```

## Comparison Methods

Each comparison method maps to specific SPAGS command-line flags:

| Method | Command Flags |
|--------|--------------|
| **R2GS** (baseline) | `--gaussiansN 1` |
| **XGS** (X-Gaussian) | Uses X-Gaussian's own `train.py` |
| **DNGS** (DNGaussian) | `--enable_depth --depth_loss_weight 0.1 --depth_loss_type pearson` |
| **CORGS** (CoR-GS) | `--gaussiansN 2 --multi_gaussian_weight 0.1` |
| **FSGS** | `--enable_fsgs_pseudo_labels --enable_fsgs_proximity_guided --enable_fsgs_complete_system` |

## Key Files

| File | Description |
|------|-------------|
| `train.py` | Main training loop with all SPAGS modules |
| `test.py` | Evaluation script (read-only) |
| `r2_gaussian/gaussian/gaussian_model.py` | Gaussian model with Student-t extension |
| `r2_gaussian/utils/loss_utils.py` | Loss functions (L1, SSIM, depth, Laplacian) |
| `r2_gaussian/utils/dngaussian_utils.py` | DNGaussian depth normalization & shape reg. |
| `r2_gaussian/utils/pseudo_view_utils.py` | FSGS pseudo-view generation (SLERP) |
| `r2_gaussian/utils/fsgs_complete.py` | FSGS complete system (proximity + depth + pseudo) |
| `r2_gaussian/utils/fsgs_proximity_optimized.py` | Proximity-guided densification |
| `r2_gaussian/arguments/__init__.py` | All command-line arguments |

## Autoresearch

SPAGS includes a Karpathy-style autonomous research loop. See `autoresearch/program_medical_gs.md` for the automated experiment protocol.

## Citation

```bibtex
@inproceedings{zha2024r2gaussian,
    title={R$^2$-Gaussian: Rectifying Radiative Gaussian Splatting for Tomographic Reconstruction},
    author={Zha, Ruyi and Zhang, Tao and Liang, Jie and Chen, Yanhao and Cai, Yuanhao and Liu, Zhiwen and Li, Zaiwen and Cao, Yanmeng and Wang, Zilong and Yu, Chenhui and Chen, Quan and others},
    booktitle={NeurIPS},
    year={2024}
}

@inproceedings{cai2024xgaussian,
    title={Radiative Gaussian Splatting for Efficient X-ray Novel View Synthesis},
    author={Cai, Yuanhao and Zha, Yixun and Wang, Jiahao and Wang, Zongwei and Chen, Yanhao and Liu, Zhiwen and Li, Zaiwen and Cao, Yanmeng and others},
    booktitle={ECCV},
    year={2024}
}
```

## Acknowledgments

This project builds upon [R²-Gaussian](https://github.com/Ruyi-Zha/r2_gaussian) and integrates innovations from [X-Gaussian](https://github.com/caiyuanhao1998/X-Gaussian), [DNGaussian](https://github.com/Fictionarry/DNGaussian), [CoR-GS](https://github.com/jiaw-z/CoR-GS), and [FSGS](https://github.com/VITA-Group/FSGS).

## License

This project is for non-commercial, research and evaluation use only. See [LICENSE.md](LICENSE.md) for details.
