#!/usr/bin/env python3
"""从eval yml提取所有75个实验的PSNR2D/SSIM2D，生成完整表格"""
import os, yaml, json

BASE = "output/baseline_full"
DATASETS = [
    'chest_50_3views',  'chest_50_6views',  'chest_50_9views',
    'foot_50_3views',   'foot_50_6views',   'foot_50_9views',
    'head_50_3views',   'head_50_6views',   'head_50_9views',
    'jaw_50_3views',    'jaw_50_6views',    'jaw_50_9views',
    'pancreas_50_3views','pancreas_50_6views','pancreas_50_9views',
]
METHODS = ['r2gaussian', 'corgs', 'fsgs', 'dngaussian', 'xgaussian']
RESULTS_FILE = "cc-agent/autoresearch/results.tsv"

def get_metrics(ds, method):
    yml_path = f"{BASE}/{ds}_{method}/eval/iter_003000/eval2d_render_test.yml"
    if not os.path.exists(yml_path):
        return None
    with open(yml_path) as f:
        d = yaml.safe_load(f)
    return d['psnr_2d'], d['ssim_2d']

# 提取所有数据
data = {}
for ds in DATASETS:
    data[ds] = {}
    for m in METHODS:
        metrics = get_metrics(ds, m)
        if metrics:
            data[ds][m] = metrics

# 打印表格 (PSNR2D)
print("=" * 120)
print("75 对比实验完整结果 — PSNR2D ↑ (测试集100张投影)")
print("=" * 120)
print(f"{'Dataset':<22}", end="")
for m in METHODS:
    print(f" {m:>12}", end="")
print(f" {'Best':>12}")
print("-" * 120)

for ds in DATASETS:
    vals = []
    best_val = -1
    best_m = ""
    for m in METHODS:
        if m in data[ds]:
            v = data[ds][m][0]
            vals.append(v)
            if v > best_val:
                best_val = v
                best_m = m
        else:
            vals.append(None)
    
    line = f"{ds:<22}"
    for v in vals:
        if v is not None:
            line += f" {v:>12.2f}"
        else:
            line += f" {'N/A':>12}"
    line += f" {best_val:>8.2f} ({best_m})"
    print(line)

print()

# 打印表格 (SSIM2D)
print("=" * 120)
print("75 对比实验完整结果 — SSIM2D ↑ (测试集100张投影)")
print("=" * 120)
print(f"{'Dataset':<22}", end="")
for m in METHODS:
    print(f" {m:>12}", end="")
print(f" {'Best':>12}")
print("-" * 120)

for ds in DATASETS:
    vals = []
    best_val = -1
    best_m = ""
    for m in METHODS:
        if m in data[ds]:
            v = data[ds][m][1]
            vals.append(v)
            if v > best_val:
                best_val = v
                best_m = m
        else:
            vals.append(None)
    
    line = f"{ds:<22}"
    for v in vals:
        if v is not None:
            line += f" {v:>12.4f}"
        else:
            line += f" {'N/A':>12}"
    line += f" {best_val:>8.4f} ({best_m})"
    print(line)

print()

# 获胜统计
print("=" * 60)
print("获胜次数统计 (PSNR2D / SSIM2D)")
print("=" * 60)
wins_psnr = {m: 0 for m in METHODS}
wins_ssim = {m: 0 for m in METHODS}
for ds in DATASETS:
    best_p = max(METHODS, key=lambda m: data[ds].get(m, (0,0))[0] if m in data[ds] else 0)
    best_s = max(METHODS, key=lambda m: data[ds].get(m, (0,0))[1] if m in data[ds] else 0)
    wins_psnr[best_p] += 1
    wins_ssim[best_s] += 1
for m in METHODS:
    print(f"  {m:>12}: PSNR {wins_psnr[m]:2d}/15  |  SSIM {wins_ssim[m]:2d}/15")

print()

# 验证测试集大小
print("=" * 60)
print("测试集验证 (100张投影)")
print("=" * 60)
for ds in ['chest_50_3views']:
    meta = json.load(open(f"data/spags_format/{ds}/meta_data.json"))
    train_angles = [round(p["angle"], 6) for p in meta["proj_train"]]
    test_angles = [round(p["angle"], 6) for p in meta["proj_test"]]
    overlap = set(train_angles) & set(test_angles)
    print(f"  {ds}")
    print(f"    训练集: {len(train_angles)} 张 (角度: {[f'{a:.2f}°' for a in train_angles]})")
    print(f"    测试集: {len(test_angles)} 张 (角度: {test_angles[0]:.2f}° ~ {test_angles[-1]:.2f}°)")
    print(f"    重叠: {'❌ ' + str(len(overlap)) if overlap else '✅ 0'}")
