---
type: concept
---
# Stable LatentMoE

## 简介

`Stable LatentMoE` 是 Kimi K3 的 sparse channel-mixing layer。它让 shared experts 在完整 model width 上处理通用变换，让 routed experts 在较窄 latent space 中处理专门变换，从而把 expert pool 与 top-k 扩大到 `896 / 16` 而不让每次路由都承担完整宽度的通信和权重流量。

## 关键属性

- 类型：latent-space Mixture-of-Experts / 极端稀疏稳定化
- K3 配置：model width 7168、latent width 3584、896 routed experts、top-16、2 shared experts
- 稳定化组件：routed aggregate 后 `RMSNorm`、`SiTU-GLU`、`Quantile Balancing`
- 稀疏比：896 个 routed experts 中激活 16 个，即 56:1 expert-pool/active ratio

## 相关主张

- LatentMoE 把 full model width 与 routed-expert width 解耦，使增大 expert pool 和 active experts 时，通信与 expert weight traffic 不必按完整 hidden dimension 成比例增长。
- 极端 sparsity 放大两类失败：连续低维 projection/GLU/up-projection 造成 internal activation explosion；近千专家的 routing 使固定步长 bias update 容易反应过慢或震荡。
- K3 在 routed expert aggregation 与 up-projection 之间加入 RMSNorm，降低不同专家和 routing weights 导致的尺度变化。
- `SiTU-GLU` 对 gate 与 up branch 分别使用 smooth tanh cap，在保留 SwiGLU 近原点形状的同时给大正激活设上界；这主要服务低精度大规模训练稳定性。
- `Quantile Balancing` 直接从当前 router margins 的分位数估计 expert bias，以目标 token load 反推下一步 dispatch threshold，替代固定步长 heuristic。
- Stable LatentMoE 不能只按“更多 experts 更强”理解；其收益依赖 expert-parallel balance、static shapes、通信 overlap 与 dedicated decode kernel 等系统条件。

## 来源支持

- [Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence](../../wiki/summaries/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.md)
- [Kimi - 2026 - Kimi K3 Open Frontier Intelligence Release](../../wiki/summaries/Kimi%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence%20Release.md)

## 关联页面

- [Kimi K3](./Kimi%20K3.md)
- [Quantile Balancing](./Quantile%20Balancing.md)
- [MoonEP](./MoonEP.md)
- [MoE](./MoE.md)
- [Muon](./Muon.md)
- [LLM 预训练](../topics/LLM%20预训练.md)
