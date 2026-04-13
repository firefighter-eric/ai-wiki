# 注意力机制 Attention

## 页面状态

- 状态：正式 topic
- 事实基座：`raw/summary/` 优先

## 主题定义

本页讨论 `attention` 作为现代 Transformer 及大模型核心算子的内部谱系，而不是把所有带“attention”字样的方法都混在一起。这里的重点是：标准 attention 如何定义；它为什么会成为统一接口；以及在面对长序列、推理带宽、显存和硬件实现约束时，后续工作如何沿着不同瓶颈分化出十多类 attention 变体。与 [Transformer](../concepts/Transformer.md) 概念页相比，本页不是解释架构名词，而是组织 attention 本身的研究主线。

## 核心问题

- 标准 `scaled dot-product attention` 为什么能成为 Transformer 的默认信息路由机制。
- `O(n^2)` 的全连接 token 交互究竟在哪些场景成为瓶颈：训练长度、推理 `KV cache`、还是硬件 IO。
- 各类 attention 变体到底在优化什么：连接图、矩阵近似、缓存结构，还是 kernel 实现。
- 哪些变体是在近似标准 attention，哪些变体则是在改写使用场景和工程接口。

## 主线脉络 / 方法分层

- 标准全连接 attention 基线：`Attention Is All You Need` 给出 `softmax(QK^T / sqrt(d_k)) V`、`multi-head attention` 与因果 / 非因果两类基本用法。它提供了统一语义，但把训练和显存成本绑定到序列长度的二次增长上。
- 长序列稀疏化路线：这条线接受“并非所有 token 对都必须显式交互”的前提，通过稀疏连接图换效率。典型形式包括：
  - 局部窗口 attention：每个 token 只看固定邻域。
  - 全局 token attention：给少量特殊 token 保留全局访问权。
  - `Longformer` 式 `local window + global attention`。
  - `BigBird` 式 `global + local + random` 混合稀疏 attention。
  - 基于内容相似性的哈希稀疏 attention，如 `Reformer` 的 `LSH attention`。
- 近似 full attention 的线性 / 低秩路线：这条线不直接规定稀疏掩码，而是尝试近似标准 attention matrix 本身。当前材料支持的代表包括：
  - 低秩投影 attention：`Linformer`。
  - 核函数线性化 attention：`Performer`。
  - landmark / Nyström 近似 attention：`Nyströmformer`。
  - 更宽泛地说，也可归入“线性 attention / low-rank attention / kernelized attention / landmark attention”等子类。
- 推理解码与 `KV cache` 优化路线：这条线关注的不是训练时的 `n x n` attention 矩阵，而是自回归解码时不断增长的 `K/V` 状态与内存带宽。典型形式包括：
  - 标准 `MHA`：每个 head 独立存 `K/V`。
  - `MQA`：多 query heads，共享单组 `K/V`。
  - `GQA`：按 query 组共享 `K/V`，在速度和质量之间折中。
  - `MLA`：对 `K/V` 做 latent 压缩，进一步降低 `KV cache`。
- 实现级 / 系统级 attention 优化路线：这条线不改变标准 attention 的数学定义，而是重写其执行方式。`FlashAttention` 代表的是 `IO-aware exact attention`，其关注点是显存访问与 kernel fusion，而不是连接模式本身。由此还可以引出：
  - exact attention kernel 优化
  - block-sparse FlashAttention
  - 面向特定硬件的 fused attention 实现

从当前证据出发，可以把常见 attention 形式至少粗分为十多种：`standard full attention`、`causal attention`、`bidirectional attention`、`multi-head attention`、局部窗口 attention、全局 token attention、随机稀疏 attention、`LSH attention`、混合稀疏 attention、低秩 attention、线性 attention、kernelized attention、Nyström attention、`MQA`、`GQA`、`MLA`、`FlashAttention` / `IO-aware exact attention`。其中有些是结构类别，有些是实现类别，不能放在同一维度上直接比较。

## 关键争论与分歧

- “高效 attention”是不是单一问题：不是。长序列训练、长文档 encoder、增量解码和 GPU kernel 优化分别对应不同瓶颈，很多论文虽然都在做 efficient attention，但优化对象并不相同。
- 稀疏 attention 与线性 attention 哪个更接近标准 attention：两者都在逼近或替代 full attention，但偏置不同。稀疏路线显式修改连接图，线性 / 低秩路线则更像近似完整矩阵。
- `MQA / GQA / MLA` 是否应与 `Linformer / Performer / BigBird` 直接并列：只能在“attention 变体”这个宽口径下并列，不能在“解决同一个瓶颈”的意义上并列。前者主要优化解码缓存，后者主要优化长序列 attention 成本。
- `FlashAttention` 是否属于新的 attention 模型：更准确地说，它是 attention 实现范式的变化，而不是新的注意力语义；它保留 exact softmax attention，只改变执行路径。
- 是否存在统一最优路线：从当前材料看，没有。不同变体在表达保真度、工程复杂度、硬件依赖和适用任务上各有边界。

## 证据基础

- [Vaswani et al. - 2017 - Attention is all you need](../../raw/summary/Vaswani%20et%20al.%20-%202017%20-%20Attention%20is%20all%20you%20need.md)：支撑标准 `scaled dot-product attention`、`multi-head attention` 与因果 / 非因果基线。
- [Kitaev, Kaiser, Levskaya - 2020 - Reformer The Efficient Transformer](../../raw/summary/Kitaev,%20Kaiser,%20Levskaya%20-%202020%20-%20Reformer%20The%20Efficient%20Transformer.md)：支撑 `LSH attention` 与哈希稀疏路线。
- [Beltagy, Peters, Cohan - 2020 - Longformer The Long-Document Transformer](../../raw/summary/Beltagy,%20Peters,%20Cohan%20-%202020%20-%20Longformer%20The%20Long-Document%20Transformer.md)：支撑局部窗口加全局 token 的长文档路线。
- [Zaheer et al. - 2020 - Big bird Transformers for longer sequences](../../raw/summary/Zaheer%20et%20al.%20-%202020%20-%20Big%20bird%20Transformers%20for%20longer%20sequences.md)：支撑混合稀疏 attention 与全局 / 局部 / 随机三元结构。
- [Wang et al. - 2020 - Linformer Self-Attention with Linear Complexity](../../raw/summary/Wang%20et%20al.%20-%202020%20-%20Linformer%20Self-Attention%20with%20Linear%20Complexity.md)：支撑低秩投影 attention。
- [Choromanski et al. - 2021 - Rethinking Attention with Performers](../../raw/summary/Choromanski%20et%20al.%20-%202021%20-%20Rethinking%20Attention%20with%20Performers.md)：支撑 kernelized / random feature 线性 attention。
- [Xiong et al. - 2021 - Nyströmformer A Nystrom-Based Algorithm for Approximating Self-Attention](../../raw/summary/Xiong%20et%20al.%20-%202021%20-%20Nystr%C3%B6mformer%20A%20Nystrom-Based%20Algorithm%20for%20Approximating%20Self-Attention.md)：支撑 landmark / Nyström 近似路线。
- [Shazeer - 2019 - Fast Transformer Decoding One Write-Head is All You Need](../../raw/summary/Shazeer%20-%202019%20-%20Fast%20Transformer%20Decoding%20One%20Write-Head%20is%20All%20You%20Need.md)：支撑 `MQA` 与解码态 `KV cache` 压缩路线。
- [Ainslie et al. - 2023 - GQA Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](../../raw/summary/Ainslie%20et%20al.%20-%202023%20-%20GQA%20Training%20Generalized%20Multi-Query%20Transformer%20Models%20from%20Multi-Head%20Checkpoints.md)：支撑 `GQA` 作为 `MHA` 与 `MQA` 之间的折中结构。
- [Unknown - 2024 - DeepSeek-V3 Technical Report](../../raw/summary/Unknown%20-%202024%20-%20DeepSeek-V3%20Technical%20Report.md)：支撑 `MLA` 作为现代 LLM 中的 latent `KV` 压缩 attention。
- [Dao et al. - 2022 - FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness](../../raw/summary/Dao%20et%20al.%20-%202022%20-%20FlashAttention%20Fast%20and%20Memory-Efficient%20Exact%20Attention%20with%20IO-Awareness.md)：支撑 `IO-aware exact attention` 与实现级优化路线。

## 代表页面

- [Transformer](../concepts/Transformer.md)
- [FlashAttention](../concepts/FlashAttention.md)
- [Grouped-Query Attention](../concepts/Grouped-Query%20Attention.md)
- [LLM 预训练](./LLM%20预训练.md)

## 未解决问题

- 当前 topic 尚未接入 `Lin et al. 2021 A Survey of Transformers` 的精修 summary，因此对更广泛 `X-former` 全景的归纳仍以代表论文为主，而非 survey 统摄。
- 旋转位置编码、相对位置偏置、`RoPE / ALiBi / YaRN` 等虽强烈影响 attention 行为，但更准确地说属于位置建模层，是否独立成 topic 仍需更多 summary 支撑。
- `Hybrid Attention`、跨模态 `cross-attention`、diffusion 中的 cross-attention、检索增强中的 chunk routing 目前只在其他 topic 零散出现，尚未并入本页。
- `MLA` 当前仅由 `DeepSeek-V3` 间接支撑；若后续要把 latent attention 单独提升为稳定概念页，应补更直接来源。

## 关联页面

- [Transformer](../concepts/Transformer.md)
- [FlashAttention](../concepts/FlashAttention.md)
- [Grouped-Query Attention](../concepts/Grouped-Query%20Attention.md)
- [LLM 预训练](./LLM%20预训练.md)
- [LLM 基础脉络](./LLM%20基础脉络.md)
