# SPAGS Autoresearch Program

这是 Karpathy's autoresearch 模式的 SPAGS 移植版。核心思想：AI agent 代替人工做实验循环——改参数、跑训练、看指标、保留或丢弃、重复。

## 实验目标

在 SPAGS（Sparse-view Prior-Augmented Gaussian Splatting）框架上，对 **ADM（Adaptive Density Modulation）模块** 进行超参数搜索，找到最优配置。

## 固定设置

每次实验使用以下固定参数：

```bash
# 数据
DATA_DIR=data/spags_format/foot_50_3views

# 基础训练参数（baseline+ADM相关）
BASIC_ARGS="--iterations 3000 --test_iterations 1000 2000 3000 --eval"
ADM_BASE="--enable_adm --adm_grid_size 256"
```

- **单次训练耗时**：约 3-5 分钟（3000 it）
- **评价指标**：`psnr_2d`（越低越好 → 越高越好！PSNR 越大越优）
- **可用数据**：foot_50_3views, foot_50_6views, foot_50_9views, head_50_3views, head_50_6views, head_50_9views

## 可修改参数

以下 ADM 超参数可以调整（**每次只改 1-2 个参数**，保持控制变量）：

| 参数 | 类型 | 默认值 | 搜索范围 | 说明 |
|------|------|--------|---------|------|
| `--adm_feat_dim` | int | 32 | [16, 32, 64, 128] | 三平面特征通道数 |
| `--adm_r_max` | float | 0.5 | [0.1, 0.3, 0.5, 0.8, 1.0] | 最大调制范围 |
| `--adm_tv_weight` | float | 0.002 | [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01] | TV正则化权重 |
| `--adm_grid_size` | int | 256 | [128, 256, 512] | 三平面网格分辨率 |

### 不能在实验中修改的

- ❌ 不能修改 `train.py` 或 `gaussian_model.py` 源码
- ❌ 不能修改 `arguments/__init__.py`
- ❌ 不能修改 `adm_module.py`
- ❌ 不能修改迭代次数 `--iterations`（保持 3000 公平比较）

## 实验步骤

### 第 0 步：检查状态

```bash
# 看之前的实验结果
ls output/
cd ~/SPAGS

# 确认数据存在
ls data/spags_format/foot_50_3views/
```

### 第 1 步：创建实验分支

```bash
git checkout -b autoresearch/adm-expr-$(date +%b%d)
```

### 第 2 步：跑 Baseline（第一次实验）

第一次实验永远先建立 baseline：

```bash
mkdir -p output/autoresearch
python train.py -s data/spags_format/foot_50_3views \
    -m output/autoresearch/baseline \
    --iterations 3000 --test_iterations 1000 2000 3000 \
    --eval > output/autoresearch/baseline.log 2>&1
```

### 第 3 步：读取结果

```bash
# 读取 PSNR 指标
cat output/autoresearch/<run_name>/eval/iter_003000/eval2d_render_test.yml

# 输出格式：
# psnr_2d: 26.85
# ssim_2d: 0.78

# 如果崩了，看日志尾巴
tail -n 30 output/autoresearch/<run_name>.log
```

### 第 4 步：记录到 results.tsv

```tsv
commit	run_name	dataset	psnr_2d	ssim_2d	adm_feat_dim	adm_r_max	adm_tv_weight	adm_grid_size	status	description
a1b2c3d	baseline	foot_50_3v	26.8518	0.7807	-	-	-	-	keep	baseline (no ADM)
b2c3d4e	feat32_r05	foot_50_3v	27.1234	0.7901	32	0.5	0.002	256	keep	ADM default params
```

### 第 5 步：迭代循环

```
LOOP FOREVER:
  1. 看 git 状态（当前分支/commit）
  2. 看 results.tsv，评估当前最佳配置
  3. 提出一个新的实验假设（看论文再看参数意义）
  4. 选择 1-2 个参数做修改
  5. 运行训练
  6. 读取 eval/iter_003000/eval2d_render_test.yml 中的 psnr_2d
  7. 如果崩了 → tail 看错误 → 修复或放弃
  8. 记录到 results.tsv（**不要 commit results.tsv**）
  9. 如果 PSNR 提升 → git commit（advance）
  10. 如果 PSNR 没提升或降低 → git reset --hard HEAD~
  11. 不要停！自动想下一个实验
```

## 实验策略建议（从易到难）

### 阶段 1：单参数扫描
先找到每个参数的敏感度：

| # | 实验 | 变量 | 固定值 | 假设 |
|---|------|------|--------|------|
| 1 | baseline | — | — | 无 ADM 的 PSNR |
| 2 | 默认 ADM | feat_dim=32, r_max=0.5, tv=0.002 | — | ADM 基本效果 |
| 3 | feat_dim 扫描 | feat_dim ∈ {16, 64} | r_max=0.5, tv=0.002 | 特征维度影响 |
| 4 | tv_weight 扫描 | tv ∈ {0.0, 0.001, 0.005} | feat_dim=32, r_max=0.5 | 正则化强度 |
| 5 | r_max 扫描 | r_max ∈ {0.3, 0.8} | feat_dim=32, tv=0.002 | 调制范围 |

### 阶段 2：最佳组合
把阶段 1 找到的最佳参数组合起来。

### 阶段 3：切换数据
用最佳参数在 head/jaw 上验证泛化性。

## 超时规则

- 单次训练最多等 **10 分钟**（3000 it 一般 3-5 分钟）
- 如果超过 10 分钟 → 视为失败，discard

## 从不停止

一旦实验循环开始，**不要停下来问人**。不要问"要继续吗？""下一步做什么？"。实验者可能在睡觉，期望你无限自主运行，直到被手动停止。如果没想法了，重新读这篇 program.md，试试组合实验策略，调更极端的参数。

每次实验约 5 分钟，每小时 ≈ 12 次实验，一晚上 ≈ 100 次实验。
