---
type: concept
---
# FlashAttention

## TL;DR（快速导读）

FlashAttention 保留标准注意力的数学结果，通过分块与减少显存读写加快执行，改变的是算子实现。

## 简介

`FlashAttention` 是标准 softmax attention 的 IO-aware 精确实现路线。它不改变 attention 的数学定义，而是通过 tile 化、kernel fusion 与更少的 HBM 读写来降低 实际用时 时间和显存压力。

## 具体怎么理解

不把完整注意力矩阵反复写入显存，而在分块计算中完成必要步骤；它与删去某些注意力连接不同。

## 关键属性

- 类型：attention 实现 / 系统优化
- 代表来源：[Dao et al. - 2022 - FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness](../../wiki/summaries/Dao%20et%20al.%20-%202022%20-%20FlashAttention%20Fast%20and%20Memory-Efficient%20Exact%20Attention%20with%20IO-Awareness.md)
- 关键区别：保持 exact attention 语义，不依赖近似 attention matrix

## 相关主张

- `FlashAttention` 证明 attention 优化不只有“改连接图 / 做近似”一条路，也可以通过重写内存访问模式获得大幅实际收益。
- 它特别重要，因为很多 attention 变体最终仍要落到 GPU kernel 与 KV/cache 读写效率问题上。

## 来源支持

- [Dao et al. - 2022 - FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness](../../wiki/summaries/Dao%20et%20al.%20-%202022%20-%20FlashAttention%20Fast%20and%20Memory-Efficient%20Exact%20Attention%20with%20IO-Awareness.md)

## 关联页面

- [Transformer](./Transformer.md)
- [注意力机制 Attention](../topics/注意力机制%20Attention.md)

## 这里的术语是什么意思

- **IO**：数据读写：计算与存储之间搬运数据的成本，可能成为速度瓶颈。
