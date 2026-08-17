---
type: concept
---
# PagedAttention

## 简介

`PagedAttention` 是 vLLM 论文提出的 KV cache 内存组织与执行机制。它借鉴操作系统分页，把一个 sequence 的逻辑 KV blocks 映射到不必物理连续的 GPU blocks，并让 attention kernel 通过 block table 读取它们。它解决的首要问题是自回归服务中 KV cache 长度动态增长、连续预留造成内部碎片和过度预留，而不是改变 Transformer attention 的数学语义。

## 关键属性

- 类型：LLM serving 的 KV cache memory management / attention execution
- 基本单位：固定 token 数的 logical KV block 与 physical KV block
- 核心映射：每个 request 维护 logical-to-physical block table，物理块可按需分配、回收和共享
- 主要收益：避免按最大生成长度预留连续显存，把单请求内部浪费限制在最后一个未填满 block，并提高可同时驻留的 request 数量
- 共享机制：不同 sequence 可引用相同 physical blocks；论文用 copy-on-write 支撑 parallel sampling、beam search 与共享 prompt
- 系统协同：vLLM 将 block-level KV manager、centralized scheduler、preemption 与 PagedAttention kernel 一起设计
- 代表来源：[Kwon et al. - 2023 - Efficient Memory Management for Large Language Model Serving with PagedAttention](../summaries/Kwon%20et%20al.%20-%202023%20-%20Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention.md)

## 相关主张

- `PagedAttention` 的核心贡献不是“让 attention 从二次复杂度变成线性”，而是把动态 KV cache 从连续 allocation 问题改写为分页 allocation 问题。
- 它与 `FlashAttention` 位于不同层：`FlashAttention` 主要优化一次 exact attention 的 HBM IO；`PagedAttention` 主要组织跨 step、跨 request 持续存在的 KV states。一个 serving engine 可以同时使用两类技术。
- `PagedAttention` 与 `RadixAttention` 也不是严格二选一。前者主要回答“KV tensors 放在哪里、怎样按块增长和共享”，后者主要回答“哪些已计算前缀值得保留、如何按 token prefix 找到并复用”；SGLang 论文明确把 RadixAttention 建立在 non-contiguous paged layout 上。
- 原始 vLLM 论文已经支持若干共享前缀场景，但当代 vLLM 又增加了 hash-based Automatic Prefix Caching。因而不能把“PagedAttention”当作当前 vLLM 的完整缓存架构。
- block size 是显存碎片、metadata / kernel 开销和共享粒度之间的折中；论文性能数字来自 2023 年模型、硬件和 baseline，不能直接代表当前框架对比。

## 来源支持

- [Kwon et al. - 2023 - Efficient Memory Management for Large Language Model Serving with PagedAttention](../summaries/Kwon%20et%20al.%20-%202023%20-%20Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention.md)
- [vLLM Project - 2026 - Automatic Prefix Caching](../summaries/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md)

## 关联页面

- [vLLM](./vLLM.md)
- [RadixAttention](./RadixAttention.md)
- [SGLang](./SGLang.md)
- [FlashAttention](./FlashAttention.md)
- [SGLang 与 vLLM 架构对比](../comparisons/SGLang%20与%20vLLM%20架构对比.md)
- [注意力机制 Attention](../topics/注意力机制%20Attention.md)
