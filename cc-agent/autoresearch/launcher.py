#!/home/qyhu/anaconda3/envs/cu116torch112_1/bin/python
"""Minimal launcher for SPAGS experiments - avoids bash entirely."""
import os, sys, subprocess

os.chdir("/home/qyhu/SPAGS")
os.environ.pop("HOME", None)  # Remove problematic HOME

gpu = sys.argv[1]
name = sys.argv[2]
extra = sys.argv[3:]

os.environ["CUDA_VISIBLE_DEVICES"] = gpu

cmd = [
    "/home/qyhu/anaconda3/envs/cu116torch112_1/bin/python",
    "train.py",
    "-s", "data/spags_format/foot_50_3views",
    "-m", f"output/autoresearch/{name}",
    "--iterations", "3000",
    "--test_iterations", "1000", "2000", "3000",
    "--eval",
] + extra

logfile = open(f"output/autoresearch/{name}.log", "w")
proc = subprocess.Popen(cmd, stdout=logfile, stderr=logfile)
sys.exit(proc.wait())
