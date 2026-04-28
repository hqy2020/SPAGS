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

### 2026-04-28 09:05 | 空间感知层次框架确立
- **背景**: 用户要求从"空间感知"出发自主研究 ADM/GAR/SPS
- **核心框架**: 层次化空间感知 (Hierarchical Spatial Awareness)
  - **SPS** (场景级): 深度归一化 → 全局3D结构理解
  - **GAR** (局部级): 邻近密化 → 几何感知的高斯分布精化
  - **ADM** (高斯级): 三平面特征网络 → 逐点密度调制
- **关键发现**: CT投影数据不含深度图，DN-Gaussian深度损失无法直接应用
- **Cron Job**: 已建立 `SPAGS每小时空间感知研究` (0 * * * *)
  - 每次只跑一个实验，自动记录、commit、push
  - 9个实验队列：单模块验证 → 两两组合 → 三合一 → 跨数据集
- **GitHub**: 已推送至 `autoresearch/adm-gradfix` 分支

### 2026-04-28 10:00 | ⚠️ 关键发现: 方法开关覆盖显式参数 — 所有实验结果需重新审视

- **背景**: Co-pruning阈值扫描(coprune_th3, coprune_th7)结果异常：PSNR=29.83/29.88 vs 预期23-24。调查发现方法开关(commit 343397c)将SPAGS模式强制设为`gaussiansN=1, coreg=False, coprune=False`
- **影响**: 即使命令显式传`--gaussiansN 2 --coprune`，`--method spags`也会覆盖为`N=1, coprune=False`。所有coprune_th3/th7实验实际上是纯ADM运行。
- **修正后的真实性能**:
  | 配置 | gaussiansN | PSNR | 提升 |
  |------|-----------|------|------|
  | 旧baseline | 2 | 27.48 | - |
  | 旧ADM feat64 | 2 | 28.90 | +1.42dB |
  | ADM-only(修正后) | **1** | **29.83** | **+2.35dB** |
- **分析**: 
  - ADM在单一高斯场(N=1)下效果更好(+2.35dB) vs 双场(N=2)的+1.42dB
  - 原因: 双高斯场的coreg_loss约束+co-pruning剪枝干扰了ADM学习
  - SPAGS方法(修正后)的admv1实际是ADM+gaussiansN=2+coreg+coprune — 完全对不齐
- **状态**: ⚠️ 需重新设计实验: (1) 公平N=1 baseline (2) 修复方法开关不覆盖显式参数 (3) 重新测试ADM+GAR+Co-pruning组合
- **决策**: 修复方法开关bug，然后跑真正干净的消融实验

### 2026-04-28 10:00 | 🔧 方法开关Bug已修复 + chest_50_3views_spags实验完成

- **Bug修复**: 移除SPAGS模式下对gaussiansN/coreg/coprune的强制覆盖。现在显式传参不再被方法开关覆盖。
- **chest_50_3views_spags结果**: PSNR=30.6222, SSIM=0.9126
  - vs chest baseline (r2gaussian, N=2): PSNR=30.5821, SSIM=0.9117
  - **提升**: +0.04dB — 几乎无改善，类似head数据集的规律
- **跨数据集ADM增益总结** (N=1 ADM vs 旧N=2 baseline):

  | 数据集 | baseline PSNR | SPAGS PSNR | 增益 | 难度 |
  |-------|--------------|-----------|------|----|
  | foot (3views) | 27.48 | 29.83 | **+2.35dB** 🔥 | 难 |
  | head (3views) | 31.17 | 31.32 | +0.14dB | 中 |
  | chest (3views) | 30.58 | 30.62 | +0.04dB | 易 |

- **分析**: ADM增益与数据集难度正相关。foot的投影角度覆盖最差→ADM空间调制效果显著；chest结构对称规则→ADM收益微乎其微
- **注意**: 以上baseline使用gaussiansN=2+coreg+coprune，不是公平的N=1对比
- **下一实验**: chest_50_6views_spags（6视角验证ADM在更多视角下的表现）

### 2026-04-28 10:14 | chest_50_6views_spags完成 — ADM在多视角下无增益

- **结果**: PSNR=33.1083, SSIM=0.9410 vs baseline (r2gaussian) PSNR=33.2370, SSIM=0.9440
- **变化**: **-0.13dB** — 6视角下ADM反而略低于baseline
- **分析**: 
  - 6视角提供足够几何约束，ADM空间调制的边际价值降至零甚至负值
  - 3视角时ADM对foot有+2.35dB增益，对chest仅+0.04dB；6视角时ADM对chest为-0.13dB
  - **核心发现**: ADM收益随视角数增加而递减，随数据难度增加而递增
- **方法对比** (chest_50_6views, PSNR2D):
  | 方法 | PSNR | 排名 |
  |-----|------|------|
  | corgs | 33.4160 | 🥇 |
  | xgaussian | 33.2663 | 🥈 |
  | dngaussian | 33.2847 | 🥉 |
  | r2gaussian | 33.2370 | 4 |
  | fsgs | 33.2028 | 5 |
|  | spags (ADM) | 33.1083 | 6 |
|- **SPAGS在6视角下排名最后** — ADM在充足视角下无优势
|
### 2026-04-28 11:00 | 实验 #X: foot_50_9views_corgs — corgs在9视角foot数据集上最优
|- **方法**: corgs (CoR-GS) on foot_50_9views
|- **结果**: PSNR2D=35.5631, SSIM2D=0.9552, PSNR3D=27.2974, SSIM3D=0.8039
|- **foot_50_9views方法排名**:
|  | 方法 | PSNR | SSIM | 排名 |
|  |-----|------|------|------|
|  | corgs | 35.5631 | 0.9552 | 🥇 |
|  | r2gaussian | 35.5573 | 0.9468 | 🥈 |
|  | fsgs | 35.4357 | 0.9482 | 🥉 |
|- **分析**: corgs在foot_50_9views上表现最佳，PSNR和SSIM均领先。corgs优势在更多视角下更明显。
||- **状态**: ✅ keep
||- **下一步**: foot_50_9views_spags (测试ADM在9视角foot上的表现)
|
|### 2026-04-28 12:00 | foot_50_9views_spags完成 — ADM在9视角foot上排名末位
||- **结果**: PSNR2D=35.3367, SSIM2D=0.9486
||- **foot_50_9views方法排名 (PSNR2D)**:
||  | 方法 | PSNR | SSIM | 排名 |
||  |-----|------|------|------|
||  | corgs | 35.5631 | 0.9552 | 🥇 |
||  | r2gaussian | 35.5573 | 0.9468 | 🥈 |
||  | xgaussian | 35.5301 | 0.9473 | 🥉 |
||  | dngaussian | 35.4420 | 0.9496 | 4 |
||  | fsgs | 35.4357 | 0.9482 | 5 |
||  | spags (ADM) | 35.3367 | 0.9486 | 6 |
||- **分析**: 与chest/head数据集的规律一致 — ADM在9视角下无优势。spags在foot_50_9views排名最后(-0.22dB vs corgs)
||- **核心发现再次确认**: ADM收益与视角数负相关，与数据难度正相关
||- **状态**: ✅ keep
||- **下一步**: 展开jaw/pancreas数据集的全方法比对实验（共6方法×6数据集=36实验）
|
|### 2026-04-28 12:00 | 实验矩阵扩展 — jaw/pancreas新数据集加入
||- **背景**: 发现jaw/pancreas数据集可用, 实验矩阵从3器官扩展到5器官
||- **新数据集**: jaw_50_3v/6v/9v, pancreas_50_3v/6v/9v (共6个新数据集)
||- **total实验**: 90 (5器官×3视角×6方法)
||- **已完成**: chest(18) + foot(18) + head(18) = 54 ✅
||- **剩余**: jaw(18) + pancreas(18) = 36
||- **jaw_50_3views_r2gaussian baseline**: PSNR2D=25.42, SSIM2D=0.8289 — 远低于其他数据集，jaw是最难的
||- **下一步**: 依次运行jaw+pancreas上全部6方法（2 GPU并行）
|
|### 2026-04-28 12:13 | jaw_50_3views全方法比对完成 — SPAGS排名第二！
||- **结果 (jaw_50_3views PSNR2D)**:
||  | 方法 | PSNR | SSIM | 排名 | 备注 |
||  |-----|------|------|------|------|
||  | corgs | 25.8061 | 0.8586 | 🥇 | 大幅领先 |
||  | spags (ADM) | 25.5676 | 0.8289 | 🥈 | +0.15dB over r2gaussian |
||  | dngaussian | 25.5135 | 0.8293 | 🥉 | +0.10dB |
||  | fsgs | 25.4330 | 0.8278 | 4 | +0.02dB |
||  | r2gaussian | 25.4160 | 0.8289 | 5 | baseline |
||  | xgaussian | 25.3936 | 0.8307 | 6 | -0.02dB |
||- **关键分析**: 
||  - jaw是迄今最难的数据集（baseline仅25.42 vs foot 27.48, head 31.17, chest 30.58）
||  - SPAGS在jaw上排名第2（+0.15dB），而在较简单的chest/head上几乎无增益
||  - **确认了ADM收益与数据难度正相关的趋势**
||  - corgs在jaw_3v上大幅领先（+0.39dB over spags），双高斯场+协同剪枝在难度数据上效果显著
|- **状态**: ✅ keep
|- **下一步**: 继续jaw_50_6views/9views + pancreas全系实验
|||
|||### 2026-04-28 13:00 | 实验 #pancreas_50_3views — corgs最佳, dngaussian/xgaussian完成
|||- **背景**: 继续pancreas_50_3views全系比对实验（共6方法）
|||- **实验结果 (pancreas_50_3views, PSNR2D)**: 
|||  | 方法 | PSNR2D | SSIM2D | 排名 |
|||  |------|--------|--------|------|
|||  | corgs | 30.6705 | 0.9203 | 🥇 |
|||  | fsgs | 30.5869 | 0.9216 | 🥈 |
|||  | r2gaussian | 30.5702 | 0.9222 | 🥉 |
|||  | xgaussian | 30.3552 | 0.9157 | 4 |
|||  | dngaussian | 30.2906 | 0.9160 | 5 |
|||  | spags (ADM) | ❓ (running) | ❓ | ❓ |
|||- **分析**:
|||  - pancreas_3v: corgs最佳(30.67), 领先fsgs/r2gaussian约+0.1dB
|||  - dngaussian(30.29)和xgaussian(30.36)明显落后, 差异约-0.3dB
|||  - 深度约束(dngaussian)在pancreas上无效（0.000000), 与CT无深度图一致
|||  - 排名与jaw_3v相似: corgs稳居第一, r2gaussian/fsgs紧追
|||- **状态**: ✅ 3/6实验完成 (corgs/dngaussian/xgaussian已记录), spags正在跑
|||- **下一步**: 完成pancreas_50_3views_spags后, 转向pancreas_50_6views全系
|||