#!/usr/bin/env python3
"""
SPAGS 每小时自主实验调度器 (v2)
覆盖全部 15 个数据集: 5器官 × 3视角
每个数据集跑 r2gaussian(baseline) + spags(ADM)
"""
import os, subprocess, re, time
from datetime import datetime

WORKDIR = os.path.expanduser("~/SPAGS")
ENV_PY = "/home/qyhu/anaconda3/envs/cu116torch112_1/bin/python"

def is_exp_done(name, results_cache=None):
    """Check if experiment is done by multiple signals"""
    # Check results.tsv
    rfile = f"{WORKDIR}/cc-agent/autoresearch/results.tsv"
    if os.path.exists(rfile):
        with open(rfile) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2 and parts[1] == name:
                    return True
    # Check log
    lfile = f"{WORKDIR}/output/autoresearch/{name}.log"
    if os.path.exists(lfile):
        with open(lfile) as f:
            if re.search(r"Training complete", f.read()):
                return True
    return False

def extract_metrics(log_path):
    if not os.path.exists(log_path): return None
    with open(log_path) as f:
        content = f.read()
    m = re.findall(r"\[ITER (\d+)\] Evaluating: psnr3d ([\d.]+), ssim3d ([\d.]+), psnr2d ([\d.]+), ssim2d ([\d.]+)", content)
    if not m: return None
    best = max(m, key=lambda x: float(x[3]))
    return float(best[3]), float(best[4]), float(best[1]), float(best[2])

def save_result(name, p2d, s2d, p3d, s3d, dataset, method):
    fname = f"{WORKDIR}/cc-agent/autoresearch/results.tsv"
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M')}\t{name}\t{p2d:.4f}\t{s2d:.4f}\t{p3d:.4f}\t{s3d:.4f}\t{dataset}\t{method}\n"
    if not os.path.exists(fname):
        with open(fname, 'w') as f:
            f.write("time\tname\tpsnr2d\tssim2d\tpsnr3d\tssim3d\tdataset\tmethod\n")
    with open(fname, 'a') as f: f.write(line)

# ──────────────────────────────────────────────
# 完整实验矩阵: 15 datasets × 2 methods
# ──────────────────────────────────────────────
ORGAN_VIEWS = [
    "chest_50_3views", "chest_50_6views", "chest_50_9views",
    "foot_50_3views",  "foot_50_6views",  "foot_50_9views",
    "head_50_3views",  "head_50_6views",  "head_50_9views",
    "jaw_50_3views",   "jaw_50_6views",   "jaw_50_9views",
    "pancreas_50_3views", "pancreas_50_6views", "pancreas_50_9views",
]

METHODS = [
    ("r2gaussian", "--method r2gaussian"),
    ("spags", "--method spags"),
]

def build_queue():
    """Build full experiment queue. Already-done experiments are excluded."""
    experiments = []
    for dv in ORGAN_VIEWS:
        for method_name, method_args in METHODS:
            exp_name = f"{dv}_{method_name}"
            cmd = (f"train.py -s data/spags_format/{dv} "
                   f"-m output/autoresearch/{exp_name} "
                   f"--iterations 3000 --test_iterations 1000 2000 3000 "
                   f"--eval {method_args}")
            experiments.append((exp_name, dv, cmd, method_name))
    return experiments

def main():
    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{t}] ⏰ SPAGS 全数据集实验调度器 (v2)")
    
    queue = build_queue()
    total = len(queue)
    print(f"  总实验数: {total} (15 datasets × 2 methods)\n")
    
    for i, (name, dataset, cmd, method) in enumerate(queue):
        if is_exp_done(name):
            print(f"  [{i+1}/{total}] ✅ {name} 已完成")
            continue
        
        print(f"  [{i+1}/{total}] ▶️ {name}")
        print(f"     dataset={dataset}, method={method}")
        
        out_dir = f"{WORKDIR}/output/autoresearch/{name}"
        log_file = f"{out_dir}.log"
        os.makedirs(out_dir, exist_ok=True)
        
        start = time.time()
        full_cmd = f"cd {WORKDIR} && {ENV_PY} {cmd} > {log_file} 2>&1"
        result = subprocess.run(full_cmd, shell=True, timeout=7200)
        elapsed = time.time() - start
        
        metrics = extract_metrics(log_file)
        if metrics:
            p2d, s2d, p3d, s3d = metrics
            save_result(name, p2d, s2d, p3d, s3d, dataset, method)
            print(f"  ✅ {name}: PSNR2D={p2d:.4f}, SSIM2D={s2d:.4f} ({elapsed:.0f}s)")
        else:
            print(f"  ❌ {name}: 无有效指标 ({elapsed:.0f}s)")
        
        # Git commit
        subprocess.run(["git", "add", "-A"], cwd=WORKDIR)
        msg = f"autoresearch: {name}"
        if metrics: msg += f" (PSNR={p2d:.2f})"
        subprocess.run(["git", "commit", "-m", msg], cwd=WORKDIR)
        try:
            subprocess.run(["git", "push", "-u", "origin", "autoresearch/adm-gradfix"], cwd=WORKDIR, timeout=30)
        except:
            pass
        
        print(f"  ⏳ 剩余 {total - i - 1} 个实验, 预计 {(total - i - 1)} 小时后完成\n")
        return  # One per invocation
    
    print("\n  🎉 全部 30 个实验已完成！下一步: 分析代表性子集")

if __name__ == "__main__":
    main()
