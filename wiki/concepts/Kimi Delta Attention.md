---
type: concept
---
# Kimi Delta Attention

## 简介

`Kimi Delta Attention (KDA)` 是 Kimi K3 大部分 attention layers 使用的 recurrent/linear attention 机制。它用固定大小状态 `S` 沿 token 序列递推，避免每层维护随序列长度增长的完整 KV cache；K3 再周期性插入 `Gated MLA`，保留不受限制的全局 token-to-token interaction。

## 关键属性

- 类型：delta-rule recurrent attention / 长上下文状态压缩
- K3 配置：69 个 KDA layers，与 24 个 Gated MLA layers 以约 `3:1` 比例混合
- 状态：每个 head 维护固定大小 recurrent matrix，而非逐 token 增长的 K/V states
- 核心改动：channel-wise forget gate、lower-bounded log-decay、full-rank output gate
- 系统实现：FlashKDA、KDA Context Parallelism、decode replay kernel、KDA-aware prefix cache

## 相关主张

- KDA 将 delta-rule update 与 channel-wise decay 组合，使不同 key channels 能以不同速率保留或遗忘历史状态。
- K3 把 log-decay 限制在 `(-5, 0)`，使 16-token tile 内的 reciprocal cumulative decay 保持在 BF16 范围，从而消除此前 diagonal tiles 的显式 position-pair path，并统一使用 Tensor Core matmul。
- KDA 的优势不是“没有状态”，而是状态大小不随序列长度增长；它把 KV capacity 问题换成 recurrent state 的串行更新、并行 scan、cache checkpoint 与回滚问题。
- `3 KDA + 1 Gated MLA` 的设计表明 KDA 并不单独承担全部全局交互：KDA 提供 recency/position-sensitive mixing，NoPE MLA 周期性恢复 unrestricted global content attention。
- 在 speculative decode 中，被拒绝 draft tokens 已经改写 recurrent state。K3 的 kernel 缓存更小的 projected inputs，并在片上重放 accepted prefix，而不是为每个 draft position 保存完整 state snapshot。
- Prefix caching 必须同时满足 MLA block 命中与 KDA boundary checkpoint 存在；因此 KDA-aware cache 不是普通 paged KV cache 的直接复用。

## 来源支持

- [Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence](../../wiki/summaries/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.md)
- [Kimi - 2026 - Kimi K3 Open Frontier Intelligence Release](../../wiki/summaries/Kimi%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence%20Release.md)

## 关联页面

- [Kimi K3](./Kimi%20K3.md)
- [Attention Residuals](./Attention%20Residuals.md)
- [PagedAttention](./PagedAttention.md)
- [vLLM](./vLLM.md)
- [SGLang](./SGLang.md)
- [注意力机制 Attention](../topics/注意力机制%20Attention.md)
