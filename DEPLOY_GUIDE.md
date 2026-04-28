# SPAGS 实验平台部署与并行训练指南

## 1. 系统要求

| 组件 | 要求 |
|------|------|
| GPU | NVIDIA GPU, ≥24GB 显存 (如 RTX 4090 / A5000 / A6000) |
| CUDA | 11.7+ |
| Python | 3.8 |
| 磁盘 | ≥100GB 可用空间 |
| 网络 | 能访问 GitHub |

## 2. 环境搭建

```bash
# 创建 conda 环境
conda create -n spags python=3.8 -y
conda activate spags

# 安装 PyTorch (CUDA 11.7)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 或 (CUDA 11.8)
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

## 3. 代码与数据部署

### 3.1 克隆代码

```bash
git clone https://github.com/hqy2020/SPAGS.git
cd SPAGS
git checkout autoresearch/adm-gradfix  # 切换到实验分支
```

### 3.2 安装子模块

```bash
# simple-knn
pip install submodules/simple-knn

# diff-gaussian-rasterization (3DGS原始)
pip install submodules/diff-gaussian-rasterization

# xray-gaussian-rasterization-voxelization (CT专用光栅化)
cd r2_gaussian/submodules/xray-gaussian-rasterization-voxelization
pip install .
cd ../../..

# 其他依赖
pip install -r requirements.txt
pip install numpy scipy pyyaml tensorboard
```

### 3.3 数据部署

**从现有服务器复制数据**（建议用 rsync/scp）：

```bash
# 在目标服务器上执行
mkdir -p data/spags_format

# 从源服务器复制 (源服务器IP假设为 192.168.1.100)
rsync -avz --progress qyhu@192.168.1.100:~/SPAGS/data/spags_format/ ./data/spags_format/

# 验证完整性
ls data/spags_format/ | wc -l  # 应该显示 15
```

15 个数据集的目录结构：
```
data/spags_format/
├── chest_50_3views/    (train=3, test=100)
├── chest_50_6views/    (train=6, test=100)
├── chest_50_9views/    (train=9, test=100)
├── foot_50_3views/     (train=3, test=100)
├── foot_50_6views/     (train=6, test=100)
├── foot_50_9views/     (train=9, test=100)
├── head_50_3views/     (train=3, test=100)
├── head_50_6views/     (train=6, test=100)
├── head_50_9views/     (train=9, test=100)
├── jaw_50_3views/      (train=3, test=100)
├── jaw_50_6views/      (train=6, test=100)
├── jaw_50_9views/      (train=9, test=100)
├── pancreas_50_3views/ (train=3, test=100)
├── pancreas_50_6views/ (train=6, test=100)
└── pancreas_50_9views/ (train=9, test=100)
```

## 4. 运行命令速查

### 4.1 单个实验

所有实验通过 `--method` 参数统一控制：

```bash
# 环境变量
PYTHON=/path/to/conda/envs/spags/bin/python

# R²-Gaussian (纯baseline)
$PYTHON train.py -s data/spags_format/chest_50_3views \
  -m output/chest_3v_r2gaussian \
  --iterations 3000 --test_iterations 1000 2000 3000 \
  --eval --method r2gaussian

# CoR-GS (双场协同)
$PYTHON train.py -s data/spags_format/foot_50_3views \
  -m output/foot_3v_corgs \
  --iterations 3000 --test_iterations 1000 2000 3000 \
  --eval --method corgs

# FSGS (邻近密化+深度)
$PYTHON train.py -s data/spags_format/head_50_3views \
  -m output/head_3v_fsgs \
  --iterations 3000 --test_iterations 1000 2000 3000 \
  --eval --method fsgs

# DN-Gaussian (深度引导)
$PYTHON train.py -s data/spags_format/jaw_50_3views \
  -m output/jaw_3v_dngaussian \
  --iterations 3000 --test_iterations 1000 2000 3000 \
  --eval --method dngaussian

# X-Gaussian (交叉视角)
$PYTHON train.py -s data/spags_format/pancreas_50_3views \
  -m output/pan_3v_xgaussian \
  --iterations 3000 --test_iterations 1000 2000 3000 \
  --eval --method xgaussian

# SPAGS (本文方法, ADM默认开启)
$PYTHON train.py -s data/spags_format/foot_50_3views \
  -m output/foot_3v_spags \
  --iterations 3000 --test_iterations 1000 2000 3000 \
  --eval --method spags
```

### 4.2 GPU 调度（多卡并行）

```bash
# GPU 0 运行
CUDA_VISIBLE_DEVICES=0 $PYTHON train.py ... --method corgs -m output/chest_3v_corgs

# GPU 1 运行 
CUDA_VISIBLE_DEVICES=1 $PYTHON train.py ... --method fsgs -m output/chest_3v_fsgs

# GPU 2 运行
CUDA_VISIBLE_DEVICES=2 $PYTHON train.py ... --method dngaussian -m output/chest_3v_dngaussian

# GPU 3 运行
CUDA_VISIBLE_DEVICES=3 $PYTHON train.py ... --method xgaussian -m output/chest_3v_xgaussian
```

### 4.3 批量运行脚本

如果目标服务器有 4 张及以上 GPU，可用以下 Python 脚本全自动并行跑：

```python
"""batch_runner.py - 4 GPU 并行批量实验"""
import os, subprocess, time
from concurrent.futures import ThreadPoolExecutor

PYTHON = "/path/to/envs/spags/bin/python"
WORKDIR = "/path/to/SPAGS"
N_GPUS = 4  # 根据实际 GPU 数量修改

DATASETS = [
    'chest_50_3views', 'chest_50_6views', 'chest_50_9views',
    'foot_50_3views',  'foot_50_6views',  'foot_50_9views',
    'head_50_3views',  'head_50_6views',  'head_50_9views',
    'jaw_50_3views',   'jaw_50_6views',   'jaw_50_9views',
    'pancreas_50_3views', 'pancreas_50_6views', 'pancreas_50_9views',
]

METHODS = ['r2gaussian', 'corgs', 'fsgs', 'dngaussian', 'xgaussian']

# 生成所有实验任务
tasks = []
for ds in DATASETS:
    for m in METHODS:
        name = f"{ds}_{m}"
        cmd = (f"CUDA_VISIBLE_DEVICES={{gpu_id}} {PYTHON} train.py "
               f"-s data/spags_format/{ds} -m output/baseline_full/{name} "
               f"--iterations 3000 --test_iterations 1000 2000 3000 "
               f"--eval --method {m}")
        tasks.append((name, cmd))

def run_task(gpu_id, name, cmd):
    full_cmd = f"cd {WORKDIR} && {cmd.format(gpu_id=gpu_id)} > output/baseline_full/{name}.log 2>&1"
    print(f"[GPU{gpu_id}] {name} 开始")
    subprocess.run(full_cmd, shell=True, timeout=7200)
    print(f"[GPU{gpu_id}] {name} 完成")

# 每 N_GPUS 个一组并行执行
for i in range(0, len(tasks), N_GPUS):
    batch = tasks[i:i+N_GPUS]
    with ThreadPoolExecutor(max_workers=N_GPUS) as pool:
        futures = []
        for j, (name, cmd) in enumerate(batch):
            future = pool.submit(run_task, j, name, cmd)
            futures.append(future)
        for f in futures:
            f.result()
    print(f"批次完成: {i+1}-{min(i+N_GPUS, len(tasks))}/{len(tasks)}")
```

> 保存为 `batch_runner.py`，运行：`python batch_runner.py`

### 4.4 双 GPU 版（已有成熟脚本）

如果只有 2 张 GPU，直接用现有脚本：

```bash
# 复制到目标服务器后运行
cd SPAGS
python cc-agent/autoresearch/run_baseline75.py
```

## 5. 结果提取

### 5.1 读取单个实验结果

```bash
# 从 yml 文件读取
cat output/baseline_full/chest_50_3views_corgs/eval/iter_003000/eval2d_render_test.yml

# 输出示例:
# psnr_2d: 30.5416
# ssim_2d: 0.9194
# psnr_2d_projs: [36.28, 36.15, ...]  (100个投影的逐张PSNR)
```

### 5.2 批量提取结果

```python
"""extract_results.py"""
import os, yaml

BASE = "output/baseline_full"
DATASETS = ['chest_50_3views', 'chest_50_6views', 'chest_50_9views',
            'foot_50_3views',  'foot_50_6views',  'foot_50_9views',
            'head_50_3views',  'head_50_6views',  'head_50_9views',
            'jaw_50_3views',   'jaw_50_6views',   'jaw_50_9views',
            'pancreas_50_3views','pancreas_50_6views','pancreas_50_9views']
METHODS = ['r2gaussian', 'corgs', 'fsgs', 'dngaussian', 'xgaussian']

print("| Dataset | R²-Gaussian | CoR-GS | FSGS | DN-Gaussian | X-Gaussian | Best |")
print("|---------|:-----------:|:------:|:----:|:-----------:|:----------:|:----:|")
for ds in DATASETS:
    vals = {}
    for m in METHODS:
        yml = f"{BASE}/{ds}_{m}/eval/iter_003000/eval2d_render_test.yml"
        if os.path.exists(yml):
            d = yaml.safe_load(open(yml))
            vals[m] = d['psnr_2d']
    best_m = max(vals, key=vals.get) if vals else ""
    best_v = vals.get(best_m, 0)
    row = f"| {ds} "
    for m in METHODS:
        v = vals.get(m, None)
        row += f"| {v:>8.2f} " if v else "|      N/A "
    row += f"| {best_v:>5.2f} ({best_m}) |"
    print(row)
```

运行: `python extract_results.py`

### 5.3 生成论文表格

```bash
cd SPAGS
python cc-agent/autoresearch/table_75.py
```

输出 Markdown 格式的完整 PSNR2D/SSIM2D 表格。

## 6. 结果同步回主服务器

```bash
# 在目标服务器上
# 压缩结果
tar -czf spags_results.tar.gz output/baseline_full/ cc-agent/autoresearch/results.tsv

# 传回源服务器
scp spags_results.tar.gz qyhu@192.168.1.100:~/SPAGS/

# 在源服务器上解压
cd ~/SPAGS
tar -xzf spags_results.tar.gz
```

## 7. 已完成的实验结果（供参考）

以下实验已在主服务器上完成，**不需要重复运行**：

### 75 Baseline 全跑完 ✅
- 全部 15 个数据集 × 5 方法 = 75 个实验
- 结果文件在 `output/baseline_full/` 和 `cc-agent/autoresearch/results.tsv`

### SPAGS (ADM) 在15个数据集上全跑完 ✅
- 结果在 `cc-agent/autoresearch/results.tsv` 中

### 建议在新服务器上运行的高优先级实验

| 优先级 | 实验 | 原因 |
|--------|------|------|
| 🔴 P0 | ADM + Co-pruning 组合 | 最高纪录 PSNR=29.25 (2000it)，需验证3000it完整收敛 |
| 🔴 P0 | SPAGS vs CoR-GS 消融 | 论文核心对比 |
| 🟡 P1 | ADM 参数调优 (r_max, feat) | 在 foot 上微调以超越 corgs 的 30.12 |
| 🟢 P2 | 可视化渲染对比 | 生成论文中的效果图 |
| 🟢 P2 | 6/9 视角跨视图分析 | 验证代表性子集结论 |

## 8. 常见问题

### Q: 训练时报 CUDA out of memory
**解决**: 减小 `--iterations` 或使用 `--test_iterations` 减少评估频率。确保 `gaussiansN=2` (CoR-GS) 时显存够用。

### Q: FSGS 报"Depth estimator not available"
**解决**: 这是已知问题，CT 数据无深度图。FSGS 会自动 fallback，不影响训练。

### Q: Docker 环境
**建议** (如果系统 CUDA 版本不兼容):
```bash
# 使用 NVIDIA PyTorch 官方镜像
docker pull nvcr.io/nvidia/pytorch:23.12-py3
docker run --gpus all -it -v /path/to/SPAGS:/workspace/SPAGS nvcr.io/nvidia/pytorch:23.12-py3 bash
cd /workspace/SPAGS
pip install -r requirements.txt
# 手动安装子模块
```

### Q: 如何确认训练/测试集无泄露
```python
import json
meta = json.load(open("data/spags_format/chest_50_3views/meta_data.json"))
train_angles = [p["angle"] for p in meta["proj_train"]]
test_angles = [p["angle"] for p in meta["proj_test"]]
overlap = set(round(a,6) for a in train_angles) & set(round(a,6) for a in test_angles)
print(f"重叠角度: {len(overlap)}")
```
