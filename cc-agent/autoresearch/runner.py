#!/usr/bin/env python3
"""
SPAGS Autoresearch Runner — 单次实验运行+结果提取

用法:
  # 跑 baseline
  python cc-agent/autoresearch/runner.py baseline

  # 跑 ADM 实验
  python cc-agent/autoresearch/runner.py run --adm_feat_dim 32 --adm_r_max 0.5 --adm_tv_weight 0.002

  # 从已有输出提取结果
  python cc-agent/autoresearch/runner.py extract output/autoresearch/my_run

  # 记录到 results.tsv
  python cc-agent/autoresearch/runner.py record --commit abc1234 --run-name feat32_r05 --psnr 27.12 --ssim 0.79 --params "feat_dim=32,r_max=0.5,tv=0.002" --status keep --desc "ADM default params"
"""

import os, sys, subprocess, json, yaml, time, argparse
from pathlib import Path

BASE_DIR = os.path.expanduser("~/SPAGS")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "autoresearch")
RESULTS_TSV = os.path.join(BASE_DIR, "cc-agent", "autoresearch", "results.tsv")

DEFAULT_DATASET = "data/spags_format/foot_50_3views"
DEFAULT_ITER = 3000

def run_baseline():
    """运行 baseline（仅 --eval, 无 ADM）"""
    run_name = "baseline"
    cmd = (
        f"cd {BASE_DIR} && python train.py "
        f"-s {DEFAULT_DATASET} "
        f"-m {OUTPUT_DIR}/{run_name} "
        f"--iterations {DEFAULT_ITER} --test_iterations 1000 2000 {DEFAULT_ITER} "
        f"--eval > {OUTPUT_DIR}/{run_name}.log 2>&1"
    )
    print(f"🚀 Running baseline: {run_name}")
    print(cmd)
    t0 = time.time()
    ret = subprocess.run(cmd, shell=True)
    elapsed = time.time() - t0
    print(f"✅ Done in {elapsed:.1f}s (exit={ret.returncode})")
    return run_name

def add_adm_args(feat_dim=32, r_max=0.5, tv_weight=0.002, grid_size=256):
    """生成 ADM 命令行参数"""
    return (f"--enable_adm --adm_grid_size {grid_size} "
            f"--adm_feat_dim {feat_dim} --adm_r_max {r_max} "
            f"--adm_tv_weight {tv_weight}")

def run_experiment(feat_dim=32, r_max=0.5, tv_weight=0.002, grid_size=256,
                   dataset=None, iterations=None):
    """运行 ADM 实验"""
    if dataset is None:
        dataset = DEFAULT_DATASET
    if iterations is None:
        iterations = DEFAULT_ITER

    # 生成过目不忘的名字
    parts = dataset.split("/")[-1]  # e.g. foot_50_3views
    run_name = f"adm_f{feat_dim}_r{r_max}_tv{tv_weight}_g{grid_size}_{parts}"

    adm_args = add_adm_args(feat_dim, r_max, tv_weight, grid_size)

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cmd = (
        f"cd {BASE_DIR} && python train.py "
        f"-s {dataset} "
        f"-m {OUTPUT_DIR}/{run_name} "
        f"--iterations {iterations} --test_iterations 1000 2000 {iterations} "
        f"--eval {adm_args} > {OUTPUT_DIR}/{run_name}.log 2>&1"
    )

    print(f"🚀 Running: {run_name}")
    print(f"   ADM: feat_dim={feat_dim}, r_max={r_max}, tv_weight={tv_weight}, grid={grid_size}")
    print(cmd)
    t0 = time.time()
    ret = subprocess.run(cmd, shell=True)
    elapsed = time.time() - t0
    print(f"✅ Done in {elapsed:.1f}s (exit={ret.returncode})")
    return run_name

def extract_results(run_path, iteration=3000):
    """从训练输出中提取 PSNR/SSIM"""
    eval_dir = os.path.join(run_path, "eval", f"iter_{iteration:06d}")
    eval_file = os.path.join(eval_dir, "eval2d_render_test.yml")

    if not os.path.exists(eval_file):
        # 试试 eval2d_render_train.yml
        eval_file = os.path.join(eval_dir, "eval2d_render_train.yml")

    if os.path.exists(eval_file):
        with open(eval_file) as f:
            data = yaml.safe_load(f)
        return {
            "psnr_2d": data.get("psnr_2d", None),
            "ssim_2d": data.get("ssim_2d", None),
        }
    else:
        return {"psnr_2d": None, "ssim_2d": None}

def record_results(commit, run_name, dataset, psnr, ssim, params, status, description):
    """记录到 results.tsv"""
    header = "commit\trun_name\tdataset\tpsnr_2d\tssim_2d\tadm_feat_dim\tadm_r_max\tadm_tv_weight\tadm_grid_size\tstatus\tdescription\n"
    line = f"{commit}\t{run_name}\t{dataset}\t{psnr}\t{ssim}\t{params}\t{status}\t{description}\n"

    if not os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV, "w") as f:
            f.write(header)

    with open(RESULTS_TSV, "a") as f:
        f.write(line)

    print(f"📝 Recorded to results.tsv: {run_name} | PSNR={psnr} | {status}")

def list_results():
    """列出所有实验结果"""
    if not os.path.exists(RESULTS_TSV):
        print("❌ results.tsv not found")
        return
    with open(RESULTS_TSV) as f:
        print(f.read())

def get_git_commit():
    """获取当前 git short hash"""
    ret = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, cwd=BASE_DIR
    )
    return ret.stdout.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPAGS Autoresearch Runner")
    subparsers = parser.add_subparsers(dest="command")

    # baseline
    p_baseline = subparsers.add_parser("baseline", help="Run baseline (no ADM)")

    # run
    p_run = subparsers.add_parser("run", help="Run ADM experiment")
    p_run.add_argument("--feat_dim", type=int, default=32)
    p_run.add_argument("--r_max", type=float, default=0.5)
    p_run.add_argument("--tv_weight", type=float, default=0.002)
    p_run.add_argument("--grid_size", type=int, default=256)
    p_run.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    p_run.add_argument("--iterations", type=int, default=DEFAULT_ITER)

    # extract
    p_extract = subparsers.add_parser("extract", help="Extract results from run")
    p_extract.add_argument("run_path", type=str)
    p_extract.add_argument("--iter", type=int, default=3000)

    # record
    p_record = subparsers.add_parser("record", help="Record results to TSV")
    p_record.add_argument("--commit", type=str, default="")
    p_record.add_argument("--run-name", type=str, required=True)
    p_record.add_argument("--dataset", type=str, default=DEFAULT_DATASET.split("/")[-1])
    p_record.add_argument("--psnr", type=float, required=True)
    p_record.add_argument("--ssim", type=float, required=True)
    p_record.add_argument("--params", type=str, default="-")
    p_record.add_argument("--status", type=str, default="keep", choices=["keep", "discard", "crash"])
    p_record.add_argument("--desc", type=str, default="")

    # list
    subparsers.add_parser("list", help="List all results")

    args = parser.parse_args()

    if args.command == "baseline":
        run_name = run_baseline()
        results = extract_results(os.path.join(OUTPUT_DIR, run_name))
        print(f"\n📊 Results: {results}")

    elif args.command == "run":
        run_name = run_experiment(
            feat_dim=args.feat_dim, r_max=args.r_max,
            tv_weight=args.tv_weight, grid_size=args.grid_size,
            dataset=args.dataset, iterations=args.iterations
        )
        results = extract_results(os.path.join(OUTPUT_DIR, run_name), args.iterations)
        print(f"\n📊 Results: {results}")

    elif args.command == "extract":
        results = extract_results(args.run_path, args.iter)
        print(yaml.dump(results, default_flow_style=False))

    elif args.command == "record":
        commit = args.commit or get_git_commit()
        # parse params into tsv columns
        params = args.params
        if params == "-":
            feat_dim, r_max, tv_weight, grid_size = "-", "-", "-", "-"
        else:
            import re
            pd = dict(item.split("=") for item in params.split(","))
            feat_dim = pd.get("feat_dim", "-")
            r_max = pd.get("r_max", "-")
            tv_weight = pd.get("tv_weight", "-")
            grid_size = pd.get("grid_size", "-")

        line = f"{commit}\t{args.run_name}\t{args.dataset}\t{args.psnr}\t{args.ssim}\t"
        line += f"{feat_dim}\t{r_max}\t{tv_weight}\t{grid_size}\t"
        line += f"{args.status}\t{args.desc}\n"

        if not os.path.exists(RESULTS_TSV):
            header = "commit\trun_name\tdataset\tpsnr_2d\tssim_2d\tadm_feat_dim\tadm_r_max\tadm_tv_weight\tadm_grid_size\tstatus\tdescription\n"
            with open(RESULTS_TSV, "w") as f:
                f.write(header)

        with open(RESULTS_TSV, "a") as f:
            f.write(line)
        print(f"📝 Recorded: {args.run_name} | PSNR={args.psnr} | {args.status}")

    elif args.command == "list":
        list_results()

    else:
        parser.print_help()
