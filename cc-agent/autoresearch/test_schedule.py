#!/usr/bin/env python3
"""Quick test of the hourly check script - dry run"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/SPAGS/cc-agent/autoresearch"))

# Test is_exp_done logic
results_file = os.path.expanduser("~/SPAGS/cc-agent/autoresearch/results.tsv")
names_to_check = ["chest_sps", "chest_gar", "chest_adm64", "chest_adm_gar", 
                  "chest_adm_gar_sps", "pancreas_baseline", "jaw_baseline",
                  "foot_adm_gar_sps", "head_adm_gar_sps"]

print("=== 实验完成状态检查 ===")
for name in names_to_check:
    # Check results.tsv
    found = False
    if os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2 and parts[1] == name:
                    found = True
                    print(f"  {'✅' if found else '❌'} {name} - 在results.tsv中")
                    break
    # Check log
    if not found:
        log_file = os.path.expanduser(f"~/SPAGS/output/autoresearch/{name}.log")
        if os.path.exists(log_file):
            with open(log_file) as f:
                content = f.read()
            if "Training complete" in content:
                print(f"  ✅ {name} - 在日志中找到 (Training complete)")
                found = True
        if not found:
            print(f"  ⬜ {name} - 待执行")

print("\n=== 下一个待执行实验 ===")
for name in names_to_check:
    log_file = os.path.expanduser(f"~/SPAGS/output/autoresearch/{name}.log")
    
    # Check results.tsv
    done = False
    if os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2 and parts[1] == name:
                    done = True
                    break
    # Check log
    if not done and os.path.exists(log_file):
        with open(log_file) as f:
            if "Training complete" in f.read():
                done = True
                
    if not done:
        print(f"  ▶️  {name}")
        break
