# Muon

## 简介

`Muon` 是当前前沿大模型中用于矩阵参数更新与近似正交化的一类优化器。当前知识库的两条直接证据分别是 `DeepSeek-V4` 的 distributed Muon 实现，以及 `Kimi K3` 针对 attention projections 引入的 `Per-Head Muon`。

## 关键属性

- 类型：LLM 训练优化器 / 大规模训练稳定性技术
- 代表来源：
  - [DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence](../../wiki/summaries/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.md)
  - [Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence](../../wiki/summaries/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.md)
- 当前角色：连接大尺度训练稳定性、矩阵正交化与 attention-head 更新平衡的优化器概念

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
