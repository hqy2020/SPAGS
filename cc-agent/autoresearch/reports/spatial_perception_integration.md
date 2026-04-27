# 空间感知（Spatial Perception）模块整合报告

## 核心发现总览

| 方法 | 核心空间感知技术 | 与 SPAGS 的关系 | 优先级 |
|------|-----------------|-----------------|--------|
| **X2-Gaussian** | 交叉注意力多视图聚合 + 3D位置预测MLP | 可增强 ADM 特征编码 | ⭐⭐⭐ |
| **DNGaussian** | 深度正则化(双向深度损失) + 深度引导初始化 | 可增强 SPS 初始化 + 几何监督 | ⭐⭐⭐⭐⭐ |
| **FSGS** | Proximity-guided密化 + 伪视图深度warp | 已部分实现(GAR)，需调优 | ⭐⭐⭐⭐ |
| **CoR-GS** | 协同剪枝(Co-pruning) + 伪视图协同正则化 | ADM 后处理补充 | ⭐⭐⭐⭐ |

## 当前关键问题

### ADM 不工作（PSNR 26.45 < baseline 27.48）
- **原因**：ADM 的梯度被 detach，只能从 TV loss 学习
- **假设**：如果能让 ADM 梯度正确流动到渲染损失，PSNR 应该提升
- **验证**：需要检查 `get_density` 和 `get_adm_density` 中的梯度流

### GAR (FSGS Proximity) 未与 ADM 组合测试
- FSGS Proximity 默认参数可能不适合 CT 场景
- proximity_threshold=6.0 可能过小或过大
- 需与 ADM 组合测试

### SPS (FDK 初始化) 完全未实现
- FDK 重建作为高斯初始位置先验
- 需要将 FDK volume 转换为点云

## 自动化探索策略

### 阶段 1: 修复 ADM（当前最重要）
1. 检查 ADM 梯度流 - 修改 get_density 使 ADM 梯度进入渲染
2. 扫描 ADM 参数（feat_dim, r_max, tv_weight）
3. 验证 ADM 修复后 PSNR 是否超过 baseline

### 阶段 2: 启用 GAR
1. 测试 FSGS Proximity 单独效果
2. 调优 proximity_threshold (4, 6, 8, 10)
3. 测试 GAR + ADM 组合

### 阶段 3: 集成外部模块
1. 从 DNGaussian 提取深度正则化
2. 从 CoR-GS 提取协同剪枝
3. 从 X2-Gaussian 提取空间注意力

### 阶段 4: 实现 SPS
1. FDK 重建管道
2. 高斯初始位置注入
3. 完整 SPS + GAR + ADM 测试
