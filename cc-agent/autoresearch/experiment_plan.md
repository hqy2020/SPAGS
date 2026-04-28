# SPAGS 标准化实验计划
## 5 个器官 × 3 个视角配置 × 2 种方法 = 30 个实验

## 实验配置
- 迭代数: 3000
- 评估节点: 1000, 2000, 3000
- 训练日志: output/{type}/{dataset}.log
- 权重: output/{type}/{dataset}/point_cloud/iteration_3000/
- 渲染结果: output/{type}/{dataset}/eval/iter_003000/render_images/
- 指标: output/{type}/{dataset}/eval/iter_003000/eval2d_render_test.yml

## 标准化输出目录结构
```
output/
├── baseline/          # Baseline (无ADM)
│   ├── chest_50_3views/
│   ├── chest_50_6views/
│   ├── chest_50_9views/
│   ├── foot_50_3views/    ... etc
│   └── pancreas_50_9views/
├── adm/               # ADM (feat_dim=64, r_max=1.0)
│   └── ...
└── ablation/          # 消融实验
    ├── adm_only/      # 只有ADM
    ├── adm_gar/       # ADM + GAR
    └── ...
```

## 批处理脚本
- run_baselines.sh: 顺序跑15个baseline
- run_adm.sh: 顺序跑15个ADM (15个基线完成后)
