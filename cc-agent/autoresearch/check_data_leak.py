#!/usr/bin/env python3
"""检查训练集和测试集角度是否重叠"""
import json, sys

for ds in ['chest_50_3views', 'chest_50_6views', 'chest_50_9views',
           'foot_50_3views', 'foot_50_6views', 'head_50_3views',
           'jaw_50_3views', 'pancreas_50_3views']:
    meta = json.load(open(f"data/spags_format/{ds}/meta_data.json"))
    train_angles = [round(p["angle"], 6) for p in meta["proj_train"]]
    test_angles = [round(p["angle"], 6) for p in meta["proj_test"]]
    
    overlap = set(train_angles) & set(test_angles)
    status = "❌ 数据泄露!" if overlap else "✅ 无重叠"
    
    print(f"{ds:25s} train={len(train_angles)} test={len(test_angles)}  {status}")
    if overlap:
        print(f"  -> 重叠角度: {sorted(overlap)}")
