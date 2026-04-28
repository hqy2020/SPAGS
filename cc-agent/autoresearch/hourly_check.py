#!/usr/bin/env python3
"""
SPAGS 每小时自主研究调度器
探索 ADM / GAR / SPS 的空间感知互补性
"""
import os, subprocess, re, time
from datetime import datetime

WORKDIR = os.path.expanduser("~/SPAGS")
ENV_PY = "/home/qyhu/anaconda3/envs/cu116torch112_1/bin/python"

def is_exp_done(name):
    """Check if experiment is already done by looking at output + result file"""
    # Check results.tsv
    rfile = f"{WORKDIR}/cc-agent/autoresearch/results.tsv"
    if os.path.exists(rfile):
        with open(rfile) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2 and parts[1] == name:
                    return True
    # Check if log exists with final metrics
    lfile = f"{WORKDIR}/output/autoresearch/{name}.log"
    if os.path.exists(lfile):
        with open(lfile) as f:
            content = f.read()
        if re.search(r"Training complete", content):
            return True
    return False

def extract_metrics(log_path):
    if not os.path.exists(log_path): return None
    with open(log_path) as f:
        content = f.read()
    m = re.findall(r"\[ITER (\d+)\] Evaluating: psnr3d ([\d.]+), ssim3d ([\d.]+), psnr2d ([\d.]+), ssim2d ([\d.]+)", content)
    if not m: return None
    best = max(m, key=lambda x: float(x[3]))  # best by psnr2d
    return float(best[3]), float(best[4]), float(best[1]), float(best[2])

def save_result(name, p2d, s2d, p3d, s3d, dataset, params):
    fname = f"{WORKDIR}/cc-agent/autoresearch/results.tsv"
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M')}\t{name}\t{p2d:.4f}\t{s2d:.4f}\t{p3d:.4f}\t{s3d:.4f}\t{dataset}\t{params}\n"
    with open(fname, 'a') as f: f.write(line)

# === Experiment Queue (priority order) ===
EXPERIMENTS = [
    ("chest_sps", "spags",
     "train.py -s data/spags_format/chest_50_3views -m output/autoresearch/chest_sps --iterations 3000 --test_iterations 1000 2000 3000 --eval --method dngaussian"),
    ("chest_gar", "spags",
     "train.py -s data/spags_format/chest_50_3views -m output/autoresearch/chest_gar --iterations 3000 --test_iterations 1000 2000 3000 --eval --method fsgs"),
    ("chest_adm64", "chest",
     "train.py -s data/spags_format/chest_50_3views -m output/autoresearch/chest_adm64 --iterations 3000 --test_iterations 1000 2000 3000 --eval --method spags"),
    ("chest_adm_gar", "spags",
     "train.py -s data/spags_format/chest_50_3views -m output/autoresearch/chest_adm_gar --iterations 3000 --test_iterations 1000 2000 3000 --eval --method spags --enable_fsgs_proximity"),
    ("chest_adm_gar_sps", "spags",
     "train.py -s data/spags_format/chest_50_3views -m output/autoresearch/chest_adm_gar_sps --iterations 3000 --test_iterations 1000 2000 3000 --eval --method spags --enable_fsgs_proximity --enable_depth --depth_loss_weight 0.04"),
    ("pancreas_baseline", "pancreas",
     "train.py -s data/spags_format/pancreas_50_3views -m output/autoresearch/pancreas_baseline --iterations 3000 --test_iterations 1000 2000 3000 --eval --method r2gaussian"),
    ("jaw_baseline", "jaw",
     "train.py -s data/spags_format/jaw_50_3views -m output/autoresearch/jaw_baseline --iterations 3000 --test_iterations 1000 2000 3000 --eval --method r2gaussian"),
    ("foot_adm_gar_sps", "foot",
     "train.py -s data/spags_format/foot_50_3views -m output/autoresearch/foot_adm_gar_sps --iterations 3000 --test_iterations 1000 2000 3000 --eval --method spags --enable_fsgs_proximity --enable_depth --depth_loss_weight 0.04"),
    ("head_adm_gar_sps", "head",
     "train.py -s data/spags_format/head_50_3views -m output/autoresearch/head_adm_gar_sps --iterations 3000 --test_iterations 1000 2000 3000 --eval --method spags --enable_fsgs_proximity --enable_depth --depth_loss_weight 0.04"),
]

def main():
    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{t}] ⏰ SPAGS 空间感知研究唤醒")
    print(f"  检查队列中 {len(EXPERIMENTS)} 个实验\n")

    for name, dataset, cmd in EXPERIMENTS:
        # Skip if done
        if is_exp_done(name):
            print(f"  ✅ {name} 已完成")
            continue
        
        print(f"  ▶️ 开始: {name}")
        print(f"     $ python {cmd}")
        
        # Prepare
        out_dir = f"{WORKDIR}/output/autoresearch/{name}"
        log_file = f"{out_dir}.log"
        os.makedirs(out_dir, exist_ok=True)
        
        # Run
        start = time.time()
        full_cmd = f"cd {WORKDIR} && {ENV_PY} {cmd} > {log_file} 2>&1"
        result = subprocess.run(full_cmd, shell=True, timeout=7200)
        elapsed = time.time() - start
        
        # Extract metrics
        metrics = extract_metrics(log_file)
        if metrics:
            p2d, s2d, p3d, s3d = metrics
            save_result(name, p2d, s2d, p3d, s3d, dataset, cmd)
            result_str = f"PSNR2D={p2d:.4f}, SSIM2D={s2d:.4f}"
            print(f"  ✅ {name}: {result_str} ({elapsed:.0f}s)")
        else:
            print(f"  ❌ {name}: 无有效指标 ({elapsed:.0f}s)")
        
        # Git commit
        subprocess.run(["git", "add", "-A"], cwd=WORKDIR)
        commit_msg = f"autoresearch: {name}"
        if metrics: commit_msg += f" (PSNR={p2d:.2f})"
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=WORKDIR)
        try:
            subprocess.run(["git", "push", "-u", "origin", "autoresearch/adm-gradfix"], cwd=WORKDIR, timeout=30)
        except:
            pass
        
        return  # One experiment per invocation
    
    print("\n  ✅ 队列中所有实验已完成！")

if __name__ == "__main__":
    main()
