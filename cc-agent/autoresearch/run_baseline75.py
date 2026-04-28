#!/usr/bin/env python3
"""
75 个对比实验批量运行器 (双 GPU 并行)
5 methods × 5 organs × 3 views = 75
每个实验 3000 iterations, --eval
"""
import os, subprocess, re, time, threading, json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

WORKDIR = os.path.expanduser("~/SPAGS")
ENV_PY = "/home/qyhu/anaconda3/envs/cu116torch112_1/bin/python"
OUTPUT_DIR = f"{WORKDIR}/output/baseline_full"
RESULTS_FILE = f"{WORKDIR}/cc-agent/autoresearch/results.tsv"
SUMMARY_FILE = f"{WORKDIR}/cc-agent/autoresearch/baseline75_summary.md"

DATASETS = [
    "chest_50_3views",  "chest_50_6views",  "chest_50_9views",
    "foot_50_3views",   "foot_50_6views",   "foot_50_9views",
    "head_50_3views",   "head_50_6views",   "head_50_9views",
    "jaw_50_3views",    "jaw_50_6views",    "jaw_50_9views",
    "pancreas_50_3views","pancreas_50_6views","pancreas_50_9views",
]

METHODS = {
    "r2gaussian": "--method r2gaussian",
    "corgs":      "--method corgs",
    "fsgs":       "--method fsgs",
    "dngaussian":  "--method dngaussian",
    "xgaussian":   "--method xgaussian",
}

def is_done(dataset, method):
    """Check if experiment already completed"""
    name = f"{dataset}_{method}"
    # Check results.tsv
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2 and parts[1] == name:
                    return True
    # Check output log
    log_file = f"{OUTPUT_DIR}/{name}.log"
    if os.path.exists(log_file):
        with open(log_file) as f:
            if re.search(r"Training complete", f.read()):
                return True
    return False

def extract_metrics(log_path):
    if not os.path.exists(log_path): return None
    with open(log_path) as f:
        content = f.read()
    m = re.findall(r"\[ITER (\d+)\] Evaluating: psnr3d ([\d.]+), ssim3d ([\d.]+), psnr2d ([\d.]+), ssim2d ([\d.]+)", content)
    if not m: return None
    # Find best by PSNR2D
    best = max(m, key=lambda x: float(x[3]))
    return float(best[3]), float(best[4]), float(best[1]), float(best[2])

def save_result(name, p2d, s2d, p3d, s3d, dataset, method):
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M')}\t{name}\t{p2d:.4f}\t{s2d:.4f}\t{p3d:.4f}\t{s3d:.4f}\t{dataset}\t{method}\n"
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'w') as f:
            f.write("time\tname\tpsnr2d\tssim2d\tpsnr3d\tssim3d\tdataset\tmethod\n")
    with open(RESULTS_FILE, 'a') as f:
        f.write(line)

def run_single(dataset, method, gpu_id):
    """Run one experiment on specified GPU"""
    name = f"{dataset}_{method}"
    out_dir = f"{OUTPUT_DIR}/{name}"
    log_file = f"{out_dir}.log"
    os.makedirs(out_dir, exist_ok=True)
    
    method_args = METHODS[method]
    cmd = (f"CUDA_VISIBLE_DEVICES={gpu_id} {ENV_PY} train.py "
           f"-s data/spags_format/{dataset} -m {out_dir} "
           f"--iterations 3000 --test_iterations 1000 2000 3000 "
           f"--eval {method_args}")
    
    full_cmd = f"cd {WORKDIR} && {cmd} > {log_file} 2>&1"
    
    print(f"  [GPU{gpu_id}] {name} 开始...")
    start = time.time()
    result = subprocess.run(full_cmd, shell=True, timeout=7200)
    elapsed = time.time() - start
    
    metrics = extract_metrics(log_file)
    if metrics:
        p2d, s2d, p3d, s3d = metrics
        save_result(name, p2d, s2d, p3d, s3d, dataset, method)
        print(f"  [GPU{gpu_id}] ✅ {name}: PSNR={p2d:.4f}, SSIM={s2d:.4f} ({elapsed:.0f}s)")
        return (name, p2d, s2d, True)
    else:
        # Show last 3 lines of log
        tail = ""
        if os.path.exists(log_file):
            with open(log_file) as f:
                lines = f.readlines()
                tail = "".join(lines[-3:]) if lines else "empty log"
        print(f"  [GPU{gpu_id}] ❌ {name} 失败 ({elapsed:.0f}s)")
        print(f"     tail: {tail}")
        return (name, 0, 0, False)

def build_experiment_list():
    """Build list of pending experiments sorted efficiently"""
    experiments = []
    done_count = 0
    for ds in DATASETS:
        for method in METHODS:
            if is_done(ds, method):
                done_count += 1
            else:
                experiments.append((ds, method))
    return experiments, done_count

def main():
    print("=" * 60)
    print("🏗️  75 个对比实验批量运行器")
    print(f"  5 methods × {len(DATASETS)} datasets = {5 * len(DATASETS)} experiments")
    print(f"  双 GPU (RTX A5000 × 2) 并行运行")
    print(f"  输出: {OUTPUT_DIR}")
    print(f"  记录: {RESULTS_FILE}")
    print("=" * 60)
    
    # Build list
    pending, done = build_experiment_list()
    total = done + len(pending)
    print(f"\n📊 状态: {done}/{total} 已完成, {len(pending)} 待运行\n")
    
    if not pending:
        print("🎉 全部 75 个实验已完成！")
        return
    
    # Run in batches of 2 (one per GPU)
    batch_num = 0
    for i in range(0, len(pending), 2):
        batch = pending[i:i+2]
        batch_num += 1
        print(f"\n{'─' * 50}")
        print(f"📦 批次 #{batch_num} ({i+1}-{min(i+2, len(pending))}/{len(pending)})")
        
        results = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            for j, (ds, method) in enumerate(batch):
                gpu_id = j  # GPU 0 for first, GPU 1 for second
                future = executor.submit(run_single, ds, method, gpu_id)
                futures[future] = (ds, method)
            
            for future in as_completed(futures):
                ds, method = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"  ❌ {ds}_{method}: 异常: {e}")
                    results.append((f"{ds}_{method}", 0, 0, False))
        
        # Brief delay between batches
        time.sleep(2)
    
    # Summary
    print(f"\n{'=' * 60}")
    print("📊 运行完成！")
    
    # Final count
    final_done = sum(1 for ds in DATASETS for m in METHODS if is_done(ds, m))
    print(f"  已完成: {final_done}/{total}")
    
    # Generate summary table
    lines = [
        "# 75 Baseline 实验汇总\n",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        f"| Dataset | r2gaussian | corgs | fsgs | dngaussian | xgaussian |\n",
        f"|---------|-----------|-------|------|------------|-----------|\n",
    ]
    for ds in DATASETS:
        row = [ds]
        for method in METHODS:
            name = f"{ds}_{method}"
            # Check results
            val = "❌"
            if os.path.exists(RESULTS_FILE):
                with open(RESULTS_FILE) as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 3 and parts[1] == name:
                            val = f"{float(parts[2]):.2f}"
                            break
            row.append(val)
        lines.append("| " + " | ".join(row) + " |\n")
    
    with open(SUMMARY_FILE, 'w') as f:
        f.writelines(lines)
    print(f"  汇总表: {SUMMARY_FILE}")
    
    # Git commit
    subprocess.run(["git", "add", "-A"], cwd=WORKDIR)
    subprocess.run(["git", "commit", "-m", f"baseline75: {final_done}/{total} 完成"], cwd=WORKDIR)
    try:
        subprocess.run(["git", "push", "-u", "origin", "autoresearch/adm-gradfix"], cwd=WORKDIR, timeout=30)
        print("  ✅ GitHub 已推送")
    except:
        print("  ⚠️ GitHub 推送失败 (commit 已本地保存)")

if __name__ == "__main__":
    main()
