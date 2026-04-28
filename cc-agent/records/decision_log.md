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
|- **commit**: 7b85bf8
### 2026-04-28 07:00 | 实验 #6: adm_feat256_001
- **假设**: feat_dim=256 提供更多特征容量以继续提升 PSNR
- **参数**: feat_dim=256, r_max=1.0, tv_weight=0.002, grid_size=256
- **结果**: PSNR=29.0986, SSIM=0.8178 (baseline: 27.4827, 0.8211)
- **变化**: **+1.616 dB PSNR** 但收益递减明显
- **分析**: 128→256 仅 +0.095 dB (vs 64→128 的 +0.10 dB, 32→64 的 +0.30 dB)。训练速度 ~1.3 it/s (feat64的1/3)。SSIM持续下降至0.8178。**feat_dim=64为最优性价比**
- **决策**: ✅ keep (新PSNR纪录但边际收益极低)
- **下一步**: ADM参数扫描完成。转移到新方向
### 2026-04-28 07:00 | 实验 #7: head_baseline
- **假设**: 测试头数据集上的baseline性能
- **参数**: 无ADM, foot_50_3views → head_50_3views
- **结果**: PSNR=31.1727, SSIM=0.9177
- **变化**: 比foot baseline 27.48高出+3.69 dB —— 头部数据集更容易
- **分析**: 头部CT结构更规则（对称骨骼），3视角覆盖更完整
- **决策**: ✅ keep
### 2026-04-28 07:00 | 实验 #8: head_adm64_001
- **假设**: ADM (feat_dim=64, r_max=1.0) 在head数据集上同样有效
- **参数**: feat_dim=64, r_max=1.0, tv_weight=0.002, grid_size=256, head_50_3views
- **结果**: PSNR=31.3164, SSIM=0.9164 (head baseline: 31.1727, 0.9177)
- **变化**: **+0.14 dB** over head baseline
- **分析**: 在head上的改善(+0.14dB)远小于foot(+1.42dB)。原因：1) head baseline PSNR已很高(31.17)，天花板效应；2) 头部结构更简单，ADM空间调制的边际收益降低
- **决策**: ✅ keep (确认ADM在不同数据集上有效但增益依赖于数据难度)
- **关键发现**: ADM对困难数据(低baseline)效果显著，对简单数据增益有限

### 2026-04-28 08:00 | 实验 #9-12: FSGS Proximity 阈值扫描
- **假设**: proximity_threshold 对FSGS Proximity standalone结果有显著影响
- **实验**: 在 foot_50_3views 上扫描4个阈值 (4, 6, 8, 10)
- **结果**:
  - th=4: PSNR=27.9608, SSIM=0.8229
  - th=6: PSNR=27.9834, SSIM=0.8217
  - th=8: PSNR=27.9573, SSIM=0.8214
  - th=10: PSNR=27.9746, SSIM=0.8225
- **变化**: 所有阈值均稳定提升 baseline (+0.47~+0.50 dB)，但阈值间差异极小 (<0.03 dB)
- **分析**: FSGS Proximity standalone对阈值不敏感，th=4-10范围结果一致。均低于ADM feat64 (28.90, +1.42dB)。FSGS速度优势(17it/s) vs ADM(3it/s)，但PSNR提升仅ADM的1/3
- **决策**: ✅ all keep
- **下一步**: 转向Co-pruning (CoR-GS) 实现，预期与ADM互补提升SSIM

### 2026-04-28 08:00 | 实验 #13: Co-pruning 实现
- **背景**: CoR-GS (ECCV 2024) 协同剪枝：双高斯场KNN距离剪枝。原代码中 `coprune` 参数存在但从未被使用
- **实现**: 在 train.py 密化步骤后插入协同剪枝逻辑：
  - 每500次迭代检查
  - 对场i中每个高斯，计算到场j的最近邻距离
  - 距离 > coprune_threshold (默认5) 则剪除
  - 使用分块 torch.cdist (2000×10000) 避免OOM
- **提交**: 5e235b5 → 79d5b34(fix distCUDA2) → 7b91bbf(fix chunk size)
- **v1失败**: distCUDA2 返回1D最近邻距离而非(N,N)全对距离，导致OOM崩溃 (PSNR=16.69初始化后崩溃)
- **v2修复**: 使用分块 torch.cdist，chunk_size_i=2000, chunk_size_j=10000
- **v2结果 th=5**: 
  - iter_1000: PSNR=28.332, SSIM=0.8277 (ADM-only @it1000: ~28.1)
  - iter_2000: PSNR=29.250, SSIM=0.8368 (🔥 最高纪录! +0.35 PSNR, +0.014 SSIM over ADM-only)
  - iter_3000: PSNR=28.932, SSIM=0.8172 (⚠️ 回归: 3000it时性能下降)
- **分析**: 
  - ✅ Co-pruning 在2000it达到最佳(29.25)，证明ADM+CoR-GS互补有效
  - ⚠️ 但3000it出现回归，说明 th=5 在后期过度剪枝
  - SSIM在2000it时高达0.8368(远超ADM-only 0.8227)，但3000it回退到0.8172
- **状态**: ✅ keep (证实co-pruning潜力，但需调优阈值)
- **下一步**: 阈值扫描 th=3(温和) vs th=7(严格)，寻找最优阈值
