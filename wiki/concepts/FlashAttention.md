# FlashAttention

## 简介

`FlashAttention` 是标准 softmax attention 的 IO-aware 精确实现路线。它不改变 attention 的数学定义，而是通过 tile 化、kernel fusion 与更少的 HBM 读写来降低 wall-clock 时间和显存压力。

## 关键属性

- 类型：attention 实现 / 系统优化
- 代表来源：[Dao et al. - 2022 - FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness](../../raw/summary/Dao%20et%20al.%20-%202022%20-%20FlashAttention%20Fast%20and%20Memory-Efficient%20Exact%20Attention%20with%20IO-Awareness.md)
- 关键区别：保持 exact attention 语义，不依赖近似 attention matrix

## 相关主张

- `FlashAttention` 证明 attention 优化不只有“改连接图 / 做近似”一条路，也可以通过重写内存访问模式获得大幅实际收益。
- 它特别重要，因为很多 attention 变体最终仍要落到 GPU kernel 与 KV/cache 读写效率问题上。

## 来源支持

- [Dao et al. - 2022 - FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness](../../raw/summary/Dao%20et%20al.%20-%202022%20-%20FlashAttention%20Fast%20and%20Memory-Efficient%20Exact%20Attention%20with%20IO-Awareness.md)
- [注意力机制 Attention](../topics/注意力机制%20Attention.md)

## 关联页面

- [Transformer](./Transformer.md)
- [注意力机制 Attention](../topics/注意力机制%20Attention.md)
