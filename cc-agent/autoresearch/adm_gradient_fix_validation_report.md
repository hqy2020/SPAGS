# SPAGS ADM 梯度修复方案 —— 二次验证与风险评估报告

## 一、核心结论

**梯度移除 `.detach()` 在概念上完全正确**，但直接实现的版本（`return base_density * modulation`）会触发 CUDA 自定义光栅化核的**反向传播形状冲突**，导致训练立即崩溃。实验日志 `adm_admDefault.log` 在第 0 轮迭代即报错：

```
_RasterizeGaussiansBackward returned an invalid gradient at index 2
- got [50000, 1] but expected shape compatible with [50000, 50000]
```

**当前代码中的 `get_density` 已经移除了 `.detach()`**，但该路径尚未成功跑完完整训练（`adm_gradfix_001` 中途中断），而所有跑完的使用 `.detach()` 版本的实验，`get_adm_density()` 的分离损失路径可能不足以让 ADM 充分学习空间调制。

---

## 二、概念验证 ✅

**修复方向是正确的**。ADM 原本仅靠 Tri-Plane 的 TV loss 学习（无渲染梯度），这导致：

- 三平面特征在 TV 正则下趋向平滑 → 调制信号趋向均匀 → ADM 退化为接近常数的乘法
- 解耦了 ADM 的优化目标与重建质量目标

移除 `.detach()` 后，ADM 可以接收来自渲染器的梯度信号，共同优化 `(base_density * modulation)` 这一整体密度值，使 ADM 学会在**需要的位置增强/削弱密度**。

---

## 三、关键风险 🔴（严重）

### 风险 1：CUDA Kernel 形状冲突（已证实）

| 实验 | 代码 | 结果 |
|------|------|------|
| `adm_default` | 有 `.detach()` | 顺利完成，PSNR2D=27.97 |
| `adm_admDefault` | 无 `.detach()` | **立即崩溃**，CUDA 反向传播形状错误 |
| `adm_phase1` | 有 `.detach()` | 顺利完成，PSNR2D=26.43 |
| `adm_tv0.001` | 有 `.detach()` | 顺利完成，PSNR2D=26.45 |
| `adm_gradfix_001` | 无 `.detach()` | 运行至约 760/3000 步后中断 |

**根因分析**：自定义 CUDA rasterizer/voxelizer 的反向函数（`_RasterizeGaussiansBackward`）在第 2 个输入（密度/不透明度）的梯度计算上做了形状约束。当 `base_density` 处于计算图中（无 `.detach()`），反向传播需要返回该输入的梯度，但 kernel 内部假设密度梯度形状为 `[N, N]` 或 `[N]`，而实际传入的 `[N, 1]` 不匹配。

### 风险 2：`[N, N]` 可能性暗示广播操作

`[N, N]` 形状的梯度通常对应矩阵乘法或广播中的雅可比累积。这意味着光栅化核可能在内部对密度做了某种广播乘法，当密度参与计算图时，反向传播产生了意外的雅可比维度膨胀。

### 风险 3：梯度爆炸 / 训练不稳定

即使修复形状问题，ADM 模块（三平面 256² × 32 通道 + MLP 解码器）在初期接受到来自渲染器的强梯度信号后，可能在 warmup 阶段产生过大的调制变化，破坏刚初始化的高斯密度场。

---

## 四、`adm_r_max=0.5` 分析

当前 `r_max=0.5` 意味着密度调制范围为 `[0.5, 1.5]` 倍。

| `r_max` | 调制范围 | 风险 |
|---------|---------|------|
| 0.1（小） | [0.9, 1.1] | 安全但效果有限 |
| 0.5（当前） | [0.5, 1.5] | 较大，初始阶段可能造成密度剧烈跳动 |
| 1.0（大） | [0, 2.0] | 密度可变为 0（高斯消失），风险高 |

**建议**：修复阶段将 `r_max` 降低至 0.2–0.3，或让 schedule 的初始 warmup 更长（从当前 20% 延至 40%）。

---

## 五、更好的替代方案

### 方案 A：修改 CUDA Kernel（推荐但工程量大）
修改 `_RasterizeGaussiansBackward` 使其接受 `[N, 1]` 形状的密度梯度，这是最根本的修复。

### 方案 B：分离 ADM 梯度路径（次选，可快速验证）
利用已有的 `get_adm_density()`，添加一个单独的 ADM 损失项：

```python
adm_density = pc.get_adm_density()  # base_density.detach() * modulation
# 添加损失：让 ADM 调制输出的密度匹配一个目标密度图
# 或使用 auxiliary rendering pass 计算梯度
```

### 方案 C：修改 `get_density` 返回值形状
确保 `base_density * modulation` 的输出形状为 `[N]`（1D）而非 `[N, 1]`（2D）：
```python
return (base_density * modulation).squeeze(-1)  # [N, 1] → [N]
```
**注意**：此方案需配合检查 `render_query.py` 中的 unsqueeze 逻辑。

### 方案 D：使用 `torch.autograd.Function` 包装
自定义一个带有梯度裁剪的 autograd Function，限制 ADM 梯度的幅值。

---

## 六、若修复后 PSNR 仍低于 baseline 的下一步

1. **增大 TV weight**：当前 0.002 的 TV loss 占主导（绝对值 ~0.0001），几乎不约束平面平滑度。建议提高到 0.01–0.05。
2. **降低学习率**：ADM 的 Adam 学习率当前未知，建议 1e-4 起步。
3. **延后 ADM 激活**：前 10%-20% 迭代不启用 ADM（仅调 base 参数），待密度场初步收敛后再开启调制。
4. **空间约束**：在每个高斯点的 offset 上加 L1 正则，避免所有高斯同时大幅调制。
5. **查看 offset/confidence 分布**：若 `offset` 全是 ±1（tanh 饱和）或 `confidence` 全为 0 或 1（sigmoid 饱和），需调整 MLP 初始化或加 BN。
