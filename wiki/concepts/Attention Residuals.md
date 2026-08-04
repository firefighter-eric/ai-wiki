# Attention Residuals

## 简介

`Attention Residuals (AttnRes)` 把 attention 的“按内容选择历史”思想从 sequence 轴迁移到 network depth。标准 residual stream 将此前层信息逐层累积到单一状态；AttnRes 让当前层用 learned pseudo-query 对 embedding 与此前 layer/block representations 分配权重，选择性读取深度历史。

## 关键属性

- 类型：跨层信息路由 / residual connection 改造
- Full AttnRes：每层可访问所有先前 layer outputs
- Block AttnRes：先把层分块，在 block-level representations 上做跨块选择，并保留当前块 partial sum
- K3 配置：8 个 12-layer blocks，外加 embedding source，共形成 9 个 block-level sources
- 主要目标：改善深层信息流，同时把存储和 pipeline communication 从 layer 数量降到 block 数量

## 相关主张

- AttnRes 解决的是**深度轴**上的历史访问，不是 token sequence attention，也不直接减少 KV cache。
- Full AttnRes 的算术量因网络深度不足百层而可承受，但保存全部 layer outputs 的 `O(Ld)` memory 和 pipeline communication 成为实际瓶颈。
- Block AttnRes 用 block sums 把 memory/communication 降为 `O(Nd)`；K3 报告称约 8 个 blocks 能恢复大部分收益。
- 在 serving 中，AttnRes 仍需要 block-state cache 与专用 kernel；K3 用 side stream overlap、sequence-parallel materialization 和 online-softmax merge 降低 prefill/decode 开销。
- AttnRes 与 DeepSeek-V4 的 `mHC` 都修改 residual information flow，但目标与数学约束不同：前者让层选择性检索深度历史，后者约束多流 residual mapping 的信号传播稳定性，不能只因都改 residual 就视为等价方法。

## 来源支持

- [Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence](../../wiki/summaries/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.md)
- [Kimi - 2026 - Kimi K3 Open Frontier Intelligence Release](../../wiki/summaries/Kimi%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence%20Release.md)

## 关联页面

- [Kimi K3](./Kimi%20K3.md)
- [Kimi Delta Attention](./Kimi%20Delta%20Attention.md)
- [Manifold-Constrained Hyper-Connections](./Manifold-Constrained%20Hyper-Connections.md)
- [Transformer](./Transformer.md)
- [LLM 预训练](../topics/LLM%20预训练.md)
