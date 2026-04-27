# 决策日志

## 2026-04-27: ADM 模块实现决策
- **背景**: 论文中描述的 ADM（Adaptive Density Modulation）三平面特征网络尚未在代码中实现
- **决策**: 实现 ADM 模块，包含 TriPlaneFeatureNetwork + DualHeadMLPDecoder
- **方案**: 
  - 新建 `r2_gaussian/utils/adm_module.py`
  - 修改 `gaussian_model.py` 集成密度调制
  - 修改 `train.py` 添加训练流程
  - 修改 `arguments/__init__.py` 添加参数
- **参数**: grid_size=256, feat_dim=32, r_max=0.5, tv_weight=0.002
- **调度**: 三阶段（warmup 20% → full 50% → decay 30%）
- **依赖项**: 需要数据到位后验证
