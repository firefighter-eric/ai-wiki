# Muon

## 简介

`Muon` 是当前前沿大模型中用于矩阵参数更新与近似正交化的一类优化器。当前知识库的两条直接证据分别是 `DeepSeek-V4` 的 distributed Muon 实现，以及 `Kimi K3` 针对 attention projections 引入的 `Per-Head Muon`。

## 关键属性

- 类型：LLM 训练优化器 / 大规模训练稳定性技术
- 代表来源：
  - [DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence](../../wiki/summaries/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.md)
  - [Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence](../../wiki/summaries/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.md)
- 当前角色：连接大尺度训练稳定性、矩阵正交化与 attention-head 更新平衡的优化器概念

## 与 AdamW 的核心区别

在现代大模型语境中，口语所称的 `Adam` 通常实际指带 decoupled weight decay 的 `AdamW`。两者最根本的差异不是是否使用 momentum，而是优化更新的基本粒度：`AdamW` 按单个参数元素维护统计量，`Muon` 则把二维权重看成完整矩阵，并调整矩阵更新的整体几何。

### AdamW：逐元素自适应

对梯度 `G_t`，AdamW 分别维护逐元素的一阶矩和二阶矩：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)G_t,
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)G_t^2,
$$

忽略 bias correction 时，其参数更新可概括为：

$$
\Delta W_t \propto \frac{m_t}{\sqrt{v_t}+\epsilon}.
$$

因此 AdamW 会根据每个元素的历史梯度尺度分别调节有效学习率，但不会直接建模这些元素共同组成的矩阵具有哪些强、弱奇异方向。由于更新是 element-wise 的，一个大矩阵可以被切分到不同 data-parallel ranks 后分别维护和更新，这也是 AdamW 易于与 ZeRO / FSDP 组合的重要原因。

### Muon：矩阵级动量正交化

Muon 先对一个 logically independent weight matrix 累积 momentum。以 DeepSeek-V4 的实现为例：

$$
M_t = \mu M_{t-1} + G_t.
$$

随后在带 Nesterov look-ahead 的 momentum 上执行 Newton–Schulz iterations。若待处理矩阵的 SVD 为：

$$
M = U\Sigma V^\top,
$$

则近似正交化的目标是得到：

$$
\operatorname{Orthogonalize}(M) \approx UV^\top.
$$

直观上，这一步弱化了原更新矩阵不同奇异方向之间的尺度差异，避免少数特别强的方向支配整个矩阵更新。实际实现通常不会直接计算完整 SVD，而是使用若干轮适合 accelerator matrix multiplication 的 Newton–Schulz iterations；之后还可以重标定 update RMS，并像 AdamW 一样使用 decoupled weight decay。

这里被近似正交化的是 **momentum / update matrix**，而不是把训练后的模型权重强制变成正交矩阵。

### 对照表

| 维度 | AdamW | Muon |
| --- | --- | --- |
| 更新粒度 | 单个参数元素 | 完整二维权重矩阵 |
| 核心统计 | 一阶矩与逐元素二阶矩 | momentum 与矩阵级近似正交化 |
| 主要视角 | 各元素历史梯度尺度 | 更新矩阵的奇异方向与整体几何 |
| optimizer state | 通常需要一阶、二阶两个逐参数 state tensors | 核心持久状态通常是 momentum buffer；实现还需要正交化临时空间 |
| 单步主要运算 | element-wise arithmetic | 多轮矩阵乘法形式的 Newton–Schulz iterations |
| 分布式切分 | 参数矩阵可以按元素切分更新 | 通常需要完整 logically independent matrix 或专门分布式实现 |
| 参数适用范围 | 可用于矩阵、向量与标量参数 | 主要针对二维矩阵参数 |
| 当前成熟度 | 通用且生态成熟 | 在大模型预训练中快速发展，但实现和调参仍更依赖具体规模 |

## 为什么通常与 AdamW 混合使用

Muon 的矩阵几何最自然地适用于 Transformer 中的二维线性变换，例如 attention 的 Q/K/V/O projections、FFN projections 和 MoE expert matrices。Embedding、prediction head、RMSNorm scale、bias 与 gating factors 的结构或梯度形态不同，不一定适合套用同一矩阵正交化规则。

因此真实的大模型训练通常不是“Muon 或 AdamW”的二选一，而是参数分工：

- 大型二维 Transformer matrices：优先考虑 Muon。
- Embedding、normalization weights、bias、部分 gating factors 与 prediction head：继续使用 AdamW。

DeepSeek-V4 正是这种混合配置。它对多数模块使用 Muon，但对 embedding、prediction head、mHC 的静态 bias / gating factors 与 RMSNorm 权重保留 AdamW。这个设计也说明，Muon 更像是针对矩阵参数的专用优化路径，而不是 AdamW 的全参数 drop-in replacement。

## Kimi K3 的 Per-Head Muon

Kimi K3 进一步指出：即使都属于二维 attention projections，把拼接后的完整 Q/K/V matrix 当作一个耦合块处理，也可能让梯度或 momentum 尺度较大的 attention heads 主导共同更新方向。

`Per-Head Muon` 因而先沿 head dimension 切分 Q/K/V momentum matrices，再对每个 head block 独立执行 Newton–Schulz orthogonalization。其目标是让不同 heads 获得更平衡的更新尺度；由于每个 tall per-head block 小于完整 projection matrix，官方报告还称该做法能略微降低 optimizer overhead。

## 工程收益与边界

- **潜在收益**：在 Transformer 预训练中，矩阵级方向平衡可能改善收敛速度与大尺度训练稳定性；DeepSeek-V4 与 Kimi K3 都把它作为联合训练 recipe 的组成部分。
- **单步成本**：Muon 需要额外的 matrix multiplications，因此“更快收敛”不等于每个 optimizer step 都更便宜；应比较达到目标 loss 的端到端训练成本。
- **状态与临时空间**：Muon 通常少于 AdamW 的两个持久 moment tensors，但需要正交化计算和临时 buffers，实际显存收益取决于精度、分片与实现。
- **分布式复杂度**：Muon 需要完整矩阵更新，与 AdamW 的 element-wise ZeRO partitioning 存在结构冲突；DeepSeek-V4 为此专门设计 bucket assignment、受限 ZeRO groups、冗余计算和 expert-wise update。
- **证据边界**：不能仅凭采用 Muon 就把模型能力或 scaling efficiency 的提升归因给优化器。当前报告中的架构、数据、模型规模与训练 recipe 同时变化，仍需要受控消融和独立复现。

一个简化但有用的直觉是：AdamW 像是逐像素调整更新幅度，Muon 则尝试校正整个矩阵更新的主方向和尺度。前者通用、易于分布式切分，后者更贴合大型线性变换的矩阵结构。

## 相关主张

- `DeepSeek-V4` 使用 `Muon` 更新多数模块，同时保留 `AdamW` 用于 embedding、prediction head、`mHC` 的静态 bias / gating factors 和 `RMSNorm` 权重。
- 官方报告把 `Muon` 的作用概括为更快收敛与更强训练稳定性，并使用 hybrid Newton-Schulz iterations 做近似正交化。
- `Muon` 在 DeepSeek-V4 中不是单独的模型能力来源，而是与 `MoE`、`CSA / HCA`、`mHC` 一起构成大规模训练和长上下文效率架构的工程底座。
- 由于 `Muon` 需要完整梯度矩阵，官方报告还专门设计了与 `ZeRO` 并行和 MoE 参数更新兼容的实现策略。
- `Kimi K3` 延续使用 Muon 更新 matrix parameters，但对 Q/K/V projections 按 attention head 切分 momentum matrices，并对每个 head block 分别执行 Newton–Schulz orthogonalization。
- Per-Head Muon 的动机是避免 full-matrix orthogonalization 将所有 heads 视为一个耦合块，导致大梯度 head 主导共同更新尺度；官方报告称该方法使 head 间学习更平衡，并略降正交化开销。
- DeepSeek-V4 与 Kimi K3 的实现侧重点不同：前者强调分布式 optimizer state、P2P gather 与 MoE/ZeRO 兼容，后者强调 attention head 内部的 block-wise update geometry。

## 来源支持

- [DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence](../../wiki/summaries/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.md)
- [Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence](../../wiki/summaries/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.md)

## 关联页面

- [DeepSeek 系列](../topics/DeepSeek%20系列.md)
- [DeepSeek-V4](./DeepSeek-V4.md)
- [Kimi K3](./Kimi%20K3.md)
- [Stable LatentMoE](./Stable%20LatentMoE.md)
- [LLM 预训练](../topics/LLM%20预训练.md)
