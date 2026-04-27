# 决策日志 — SPAGS 空间感知自主研究

## 实验记录格式
每条记录包含：
```
YYYY-MM-DD HH:MM | exp-<name> | PSNR=XX.XXXX/SSIM=XX.XXXX | 参数: ... | 状态: keep/reject | 分析: ...
```

## 历史记录

### 2026-04-27: ADM 模块实现决策
- **背景**: 论文中描述的 ADM（Adaptive Density Modulation）三平面特征网络尚未在代码中实现
- **决策**: 实现 ADM 模块，包含 TriPlaneFeatureNetwork + DualHeadMLPDecoder
- **方案**: 
  - 新建 `r2_gaussian/utils/adm_module.py`
  - 修改 `gaussian_model.py` 集成密度调制
  - 修改 `train.py` 添加训练流程
  - 修改 `arguments/__init__.py` 添加参数
- **参数**: grid_size=256, feat_dim=32, r_max=0.5, tv_weight=0.002
- **调度**: 三阶段（warmup 20% → full 50% → decay 30%）
- **结果**: PSNR=26.43 (低于 baseline 27.48) — ADM 梯度被 detach
- **分析**: ADM 仅通过 TV loss 学习，无法从 rendering loss 获得有效梯度

### 2026-04-28: 空间感知自主研究体系搭建
- **背景**: 四个相关论文调研完成（X2-Gaussian, DNGaussian, FSGS, CoR-GS）
- **决策**: 搭建每小时自动运行的自主研究系统
- **方案**: 
  - 创建 skill `spags-spatial-perception-research`
  - 创建 cron job（每小时 0 分执行）
  - 每次实验自动记录到 results.tsv 和 decision_log.md
  - 每次实验后 git commit + push 到 GitHub
- **报告位置**: `cc-agent/autoresearch/reports/`
### 2026-04-28 06:00 | 实验 #1: adm_gradfix_001
- **假设**: 移除 get_density 的 .detach() 让 ADM 梯度通过 rasterizer 反向传播
- **修改**: r2_gaussian/gaussian/gaussian_model.py → get_density() 去掉所有 .detach()
- **结果**: PSNR=28.2981, SSIM=0.8172 (baseline: 27.4827, 0.8211)
- **变化**: **+0.8154 dB PSNR! 首次超越 baseline**
- **状态**: ✅ keep
- **分析**: ADM 梯度修复完全正确。三平面特征网络现在能从渲染损失学习，TV loss 作为辅助正则化。iter 1000 时 PSNR=27.5380 已超 baseline。
### 2026-04-28 06:00 | 实验 #5: adm_feat128_001
- **假设**: 更多特征维度 (feat_dim=128) 能继续提升 PSNR
- **参数**: feat_dim=128, r_max=1.0, tv_weight=0.002, grid_size=256
- **结果**: PSNR=29.0035, SSIM=0.8181 (baseline: 27.4827, 0.8211)
- **变化**: **+1.5208 dB PSNR! 突破29.00!** 但SSIM略降(-0.0030)
- **分析**: feat_dim 64→128 仅带来 +0.10 dB 改善（vs 32→64 的 +0.30 dB），收益递减。训练速度从~3 it/s降至~1.14 it/s（3x减慢），性价比不高。
- **决策**: ✅ keep（PSNR新纪录），但ADM参数扫描接近最优限制
- **下一步**: 转向第2优先级 GAR (FSGS Proximity) 集成
- **commit**: 7b85bf8
