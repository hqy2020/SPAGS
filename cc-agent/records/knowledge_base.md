# 知识库 (Knowledge Base)

> 本文档记录科研助手团队在项目过程中积累的所有知识、经验和教训

---

## 📚 索引

- [成功案例](#成功案例)
- [失败教训](#失败教训)
- [技术知识点](#技术知识点)
- [医学领域知识](#医学领域知识)
- [最佳实践](#最佳实践)

---

## ✅ 成功案例

### ADM在双高斯场(N=2) vs 单场(N=1) — 2026-04-28  
**发现**: ADM在N=1下 PSNR=29.83(+2.35dB) 显著优于 N=2+coreg+coprune 的 28.90(+1.42dB)。双场的coreg_loss约束干扰了ADM学习。  
**方法开关Bug**: `--method spags` 强制覆盖gaussiansN=1，即使显式传参。已在10:00修复。

### 模板
```markdown
### [日期] 案例名称
- **创新点来源：** 论文链接/名称
- **实现方法：** 简要描述
- **性能提升：** PSNR +X dB / SSIM +X
- **关键决策：** 列出 2-3 个关键决策点
- **可复用组件：** 代码路径或方法
- **参考文档：** 相关分析报告链接
```

### 2026-04-28 ADM 梯度修复成功
- **创新点来源：** 论文 ADM 模块（Adaptive Density Modulation）梯度阻断问题修复
- **实现方法：** 移除 `get_density()` 中的 `.detach()`，使 ADM 三平面特征网络的梯度通过 CUDA rasterizer 反向传播到渲染损失
- **性能提升：** PSNR +0.82 dB（27.48 → 28.30），最终推至 +1.42 dB（28.90 with feat_dim=64, r_max=1.0）
- **关键决策：** 
  1. 发现 `.detach()` 阻断梯度流是 ADM 不工作的根本原因
  2. r_max=1.0 比 r_max=0.5 效果更好（+0.30 dB）
  3. feat_dim=64 比 feat_dim=32 效果更好（+0.30 dB）
- **可复用组件：** `r2_gaussian/utils/adm_module.py` 中的 TriPlaneFeatureNetwork + DualHeadMLPDecoder
- **参考文档：** `cc-agent/autoresearch/adm_gradient_fix_validation_report.md`
- **参数基线：** grid_size=256, feat_dim=64, r_max=1.0, tv_weight=0.002

---

### 2026-04-28 ADM feat_dim 收益递减验证
- **发现**: feat_dim=32→64: +0.30 dB, 64→128: +0.10 dB, 128→256: +0.095 dB
- **确认规律**: 收益呈对数递减。feat_dim=64 是 PSNR/速度的最佳平衡点
- **训练速度**: feat_dim=64 (~3-4 it/s), feat_dim=128 (~1.14 it/s), feat_dim=256 (~1.3 it/s)
- **SSIM变化**: feat_dim=32 (0.8172), feat_dim=64 (0.8227), feat_dim=128 (0.8181), feat_dim=256 (0.8178) — feat_dim=64 SSIM最优
- **建议**: 默认使用 feat_dim=64

### 2026-04-28 ADM 在 head 数据集上的泛化验证
- **发现**: ADM (feat_dim=64, r_max=1.0) 在 head_50_3views 上 +0.14 dB (31.17→31.32)，而在 foot 上 +1.42 dB (27.48→28.90)
- **分析**: ADM 的增益与数据难度负相关。简单数据集（高 baseline PSNR）上增益小，困难数据集上增益大
- **可能原因**: (1) 天花板效应 - head 数据集 PSNR 已接近 31； (2) 头部结构更简单、对称，ADM 的空间调制优势不显著
- **建议**: 在多数据难度谱上验证，对容易数据集降低 ADM 强度

### 2026-04-28 在多个数据集上并行实验
- **发现**: GPU 0 + GPU 1 可并行跑两个实验，显著提高吞吐量
- **方法**: CUDA_VISIBLE_DEVICES=1 分配到空闲 GPU
- **建议**: 有2个 GPU 时，主实验在 GPU 0 跑，辅助实验（baseline、新数据验证）在 GPU 1 跑

### 2026-04-28 Co-pruning (CoR-GS) 成功集成
- **创新点来源：** CoR-GS (ECCV 2024) 协同剪枝：双高斯场结构一致性剪枝
- **实现方法：** 在 train.py 密化步骤后每500迭代执行双场KNN剪枝，使用分块 torch.cdist (2000×10000) 计算跨场距离
- **关键实现细节：**
  - ⚠️ `distCUDA2` 不可用（返回1D最近邻距离，非(N,N)全对距离）
  - ⚠️ broadcast 不可用（100k点下需要120GB显存）
  - ✅ 正确做法：分块 torch.cdist, chunk_size_i=2000, chunk_size_j=10000
- **性能提升（th=5）：** iter_2000: PSNR=29.25 (+0.35 over ADM-only 28.90), SSIM=0.8368 (+0.014)
- **重要发现：** Co-pruning th=5 在 iter_2000 达到最佳，但 iter_3000 出现回归 (28.93/0.8172)，说明默认阈值5在后期过度剪枝
- **待调优：** 需要阈值扫描 (th=3, 7, 10) 找到最优剪枝力度
- **提交：** 5e235b5 → 79d5b34 → 7b91bbf

---

## ❌ 失败教训

### 模板
```markdown
### [日期] 失败案例名称
- **尝试目标：** 想要实现什么
- **失败原因：** 根本原因分析
- **错误假设：** 事后发现的错误假设
- **性能影响：** PSNR -X dB 或其他指标下降
- **教训总结：** 下次如何避免
- **参考文档：** result_analysis.md 路径
```

### 2026-04-28 ADM r_max=1.5 导致 NaN 崩溃
- **尝试目标：** 探索更大的调制范围 r_max=1.5 是否能进一步提升 PSNR
- **失败原因：** 调制过强（r_max=1.5）导致某些区域的密度变为负值，数值不稳定
- **错误假设：** 认为 r_max=1.0→1.5 的线性扩展能继续提升
- **性能影响：** NaN（完全崩溃）
- **教训总结：** CT 密度值范围有限（~0.05-0.5），r_max 不应超过 1.0-1.2 范围
- **参考文档：** `output/autoresearch/adm_rmax1.5_001.log`

### 2026-04-28 ADM TV loss 在梯度阻断时无效
- **尝试目标：** 通过调整 TV loss 权重使 ADM 学习
- **失败原因：** `.detach()` 阻断 ADM 梯度，ADM 无法从渲染损失或 TV loss 学习有效信号
- **错误假设：** 认为 TV 正则化能单独驱动 ADM 训练
- **性能影响：** PSNR=26.43-26.45（低于 baseline 27.48）
- **教训总结：** 任何形式的 ADM 训练都必须有梯度流通过渲染损失；TV loss 在梯度修复后才生效

### 2026-04-28 Co-pruning 默认阈值 th=5 在 3000it 过度剪枝
- **尝试目标：** 使用 CoR-GS 默认阈值 th=5 剪枝不稳定高斯
- **失败原因：** th=5 在训练后期(>2000it)过于激进，剪除了过多有用高斯
- **表现：** iter_2000 达到峰值 29.25/0.8368，但 iter_3000 回退到 28.93/0.8172
- **教训总结：** co-pruning 阈值需要针对 3000it 短训练任务调优，默认 th=5 适合长训练(30000it)但短训练下需更温和
- **参考文档：** `output/autoresearch/adm_feat64_coprune_v2/eval/iter_003000/eval2d_render_test.yml`

---

## 🧠 技术知识点

### 3D Gaussian Splatting 相关
- **球谐函数 (Spherical Harmonics)：**
  - 用于表示视角相关的颜色变化
  - 阶数越高表达能力越强，但计算开销增加

- **自适应密度控制：**
  - Clone：分裂大梯度区域的 Gaussians
  - Prune：移除透明度低的 Gaussians
  - 典型阈值：opacity < 0.005

### R²-Gaussian 特有技术

- **X 射线投影 vs RGB 渲染：**
  - R²-Gaussian 使用穿透式投影（累积密度），而非标准 3DGS 的 alpha 混合
  - 自定义 CUDA 算子：`xray-gaussian-rasterization-voxelization`
  - 支持锥束和平行束几何

- **坐标归一化：**
  - 整个场景（扫描仪 + CT 体）归一化到 [-1,1]³ 空间
  - 高斯尺度参数化为体积百分比（避免绝对尺度导致的数值不稳定）
  - 关键参数：`scale_min=0.0005`, `scale_max=0.5`

- **FDK 初始化：**
  - 使用 TIGRE 工具箱的 FDK 算法从稀疏投影重建初始体数据
  - 采样高密度区域（`density_thresh=0.05`）作为初始点云
  - 密度缩放因子（`density_rescale=0.15`）补偿遮挡效应
  - **初始化质量直接决定成败** - 建议用 `--evaluate` 检查初始 PSNR

- **密化策略：**
  - 基于梯度的分裂：`densify_grad_threshold=0.00005`
  - 基于尺度的分裂：`densify_scale_threshold=0.1`（体积的 10%）
  - 剪枝阈值：`density_min_threshold=0.00001`
  - 密化周期：每 100 次迭代，从 500 迭代开始，到 15,000 迭代结束

- **损失函数组合：**
  - L1 + SSIM（`lambda_dssim=0.25`）用于 2D 投影监督
  - TV 正则化（`lambda_tv=0.05`）用于 3D 体平滑
  - 可选：深度损失、光度一致性损失（IPSM）

### PyTorch 优化技巧
- **混合精度训练：** 使用 `torch.cuda.amp` 可加速 30-50%
- **梯度累积：** 当 GPU 内存不足时的有效策略
- **学习率调度：** ExponentialLR 在 3DGS 中效果优于 CosineAnnealing
- **R²-Gaussian 学习率设置：**
  - 位置：0.0002 → 0.00002（30,000 步）
  - 密度：0.01 → 0.001
  - 尺度：0.005 → 0.0005
  - 旋转：0.001 → 0.0001

### CUDA 优化经验
- **Tile-based Rasterization：** 3DGS 的核心渲染方式
- **并行排序：** 需要高效的 radix sort 实现
- **X 射线投影加速：** R²-Gaussian 的 CUDA 内核针对穿透式累积优化

### CoR-GS co-pruning 实现要点
- **核心逻辑：** 双高斯场之间KNN距离剪枝，场i中点找场j中最近邻，距离>阈值则剪除
- **⚠️ 不可用 distCUDA2：** 返回的是(到最近邻的距离, N)而非(N,N)全对距离矩阵
- **⚠️ 不可用 broadcast：** xyz_i[:,None,:] - xyz_j[None,:,:] 在100k点下需要120GB显存
- **✅ 正确实现：** 分块 torch.cdist, chunk_size_i=2000, chunk_size_j=10000
- **验证：** commit 7b91bbf 包含两次修复迭代
- **短训练注意事项：** 默认 th=5 适合长训练(30000it)，3000it 短训练下可能过度剪枝

---

## 🏥 医学领域知识

### CT 成像特性
- **Hounsfield Unit (HU)：**
  - 空气: -1000 HU
  - 水: 0 HU
  - 骨骼: +400 ~ +1000 HU

- **窗宽窗位 (Window Level/Width)：**
  - 肺窗：WL=-600, WW=1500
  - 软组织窗：WL=40, WW=400
  - 骨窗：WL=400, WW=1800

### 临床评估标准
- **诊断可用性：** 优先于纯数值指标（PSNR/SSIM）
- **伪影类型：**
  - 金属伪影：高密度物体周围的条纹
  - 运动伪影：患者移动导致的模糊
  - 噪声：低剂量扫描的颗粒感

### 稀疏视角挑战
- **角度欠采样：** 少于 180° 覆盖会导致严重伪影
- **投影数不足：** 通常需要 ≥180 个投影，稀疏场景可能仅 30-60 个

---

## 🎯 最佳实践

### 实验设计
1. **消融实验原则：** 每次只改变一个变量
2. **基线对比：** 始终与原始 baseline 和 SOTA 方法对比
3. **多数据集验证：** 至少在 2-3 个数据集上验证泛化性

### 代码管理
1. **分支策略：**
   - `main`：稳定 baseline
   - `dev-feature-name`：新功能开发
   - 合并前必须通过测试

2. **提交规范：**
   ```
   [角色] 简要描述

   - 详细说明修改内容
   - 关联的 issue/实验编号
   ```

3. **配置文件版本化：** 所有超参数用 YAML/JSON 管理，避免硬编码

### 文档维护
1. **及时记录：** 完成任务后立即更新 record.md
2. **交叉引用：** 使用相对路径链接相关文档
3. **定期归档：** 每月整理 records/ 下的临时文件

---

## 🔗 快速链接

- [决策日志](./decision_log.md) - 查看历史决策
- [项目时间线](./project_timeline.md) - 查看进度
- [3DGS 专家分析报告](../3dgs_expert/analyses/)
- [实验结果汇总](../experiments/results/)

---

## 🔧 工具配置经验

### MCP 服务器安装

**必需工具：**
1. **@modelcontextprotocol/server-arxiv** - 论文搜索下载
2. **@modelcontextprotocol/server-github** - 代码调研（需 GitHub Token）
3. **@modelcontextprotocol/server-filesystem** - 文件系统访问
4. **@modelcontextprotocol/server-sqlite** - 实验数据库
5. **@modelcontextprotocol/server-brave-search** - 网络搜索（可选）

**配置位置：**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%/Claude/claude_desktop_config.json`

**API 密钥获取：**
- GitHub Token: https://github.com/settings/tokens（需 `repo` 权限）
- Brave Search: https://brave.com/search/api/（免费 2000 次/月）

**验证方法：**
```
重启 Claude Desktop 后，尝试："请使用 arXiv 工具搜索 3D Gaussian Splatting"
```

### 环境安装踩坑记录

**TIGRE 安装问题：**
- 可能需要 `--no-build-isolation` 标志
- 需要先安装 Cython：`pip install Cython==0.29.36`
- Ubuntu 需要 gcc/g++ 编译器

**CUDA 扩展编译失败：**
- 检查 CUDA 版本匹配：`nvcc --version` 应为 11.6
- 确保 PyTorch CUDA 版本一致：`torch.version.cuda`
- 子模块未初始化：`git submodule update --init --recursive`

**常见错误：**
```bash
# 错误：ModuleNotFoundError: No module named 'simple_knn._C'
# 解决：重新安装子模块
cd r2_gaussian/submodules/simple-knn
pip install -e .

# 错误：xray_gaussian_rasterization_voxelization 编译失败
# 解决：检查 CUDA_HOME 环境变量
export CUDA_HOME=/usr/local/cuda-11.6
```

---

**最后更新：** 2026-04-28
**维护者：** 进度跟踪与协调秘书
