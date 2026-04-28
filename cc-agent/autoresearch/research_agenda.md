# SPAGS 空间感知研究议程

## 核心目标
从稀疏视角CT重建的**空间感知**出发，探索 ADM / GAR / SPS 三大模块的创新设计。

## 三大核心模块

### ADM - Adaptive Density Modulation ✅ (已实现)
- **功能**: 三平面特征网络预测每个高斯的密度调制因子
- **空间感知机制**: 通过可学习的空间特征编码，在不同区域自适应调节高斯密度
- **当前状态**: 已实现，feat_dim=64最优，+1.42dB PSNR提升
- **待探索**: 与GAR/SPS的互补效应

### GAR - Geometry-Aware Regularization (探索中)
- **候选定义**:
  1. Gradient-Aware Refinement — 梯度感知精化
  2. Geometry-Aware Regularization — 几何感知正则化
  3. Gaussian Attention Refinement — 高斯注意力精化
- **FSGS相关性**: Proximity-guided densification (几何邻近引导密化)
- **DN-Gaussian相关性**: Depth-guided regularization (深度引导正则化)
- **待验证**: 是否可将FSGS Proximity + DN-Gaussian深度正则化为统一的GAR模块

### SPS - Spatial Perception System (探索中)
- **候选定义**:
  1. Spatial Progressive Sampling — 空间渐进采样
  2. Structure-Preserving Supervision — 结构保持监督
  3. Scene Perception System — 场景感知系统
- **X2-Gaussian相关性**: Cross-view attention (跨视角注意力)
- **待验证**: 交叉注意力是否可作为SPS的核心空间感知引擎

## 研究计划
1. 调研X2-Gaussian的交叉注意力机制
2. 调研DN-Gaussian的深度正则化机制
3. 调研FSGS的邻近密化+伪视图机制
4. 提取可迁移的空间感知模块
5. 调用实验验证互补性
6. 形成最终论文创新点
