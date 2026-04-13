# Vaswani et al. - 2017 - Attention is all you need

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Vaswani et al. - 2017 - Attention is all you need.pdf
- 原始 HTML：../../raw/html/Vaswani et al. - 2017 - Attention is all you need.html
- 全文文本：../../raw/text/Vaswani et al. - 2017 - Attention is all you need.md
- 作者：Vaswani et al.
- 年份：2017
- 状态：已基于 arXiv HTML 整理

## 自动抽取摘要或人工摘要

这篇论文的决定性意义不只是提出 `Transformer`，而是把“标准全连接自注意力”确立为序列建模的默认骨架：所有 token 两两交互，依靠 `scaled dot-product attention`、`multi-head attention` 与位置编码取代 RNN/CNN 的主导角色。后续大量 attention 变体基本都以本文的 `O(n^2)` 全注意力为参照物，要么试图近似它、稀疏化它，要么在不改变其语义的前提下优化实现。

## 关键事实

- 论文将标准 attention 明确定义为 `softmax(QK^T / sqrt(d_k)) V` 的缩放点积形式，并把它作为 Transformer 的核心算子。
- `multi-head attention` 的作用不是简单并行重复，而是让模型在不同子空间、不同位置关系上同时建模依赖。
- 编码器使用双向 self-attention，解码器使用带因果 mask 的 self-attention，说明“attention 变体”从一开始就同时包含双向与自回归两类用法。
- 本文的全连接 attention 具有全局感受野和强表达力，但时间与显存开销都随序列长度呈二次增长，后续高效 attention 工作几乎都围绕这一瓶颈展开。
- 从知识组织角度看，`standard attention` 不是“十多种 attention 中的一种小变体”，而是后续线性、稀疏、低秩、IO-aware 与 KV-cache 优化路线的共同基线。

## 争议与不确定点

- 本文解决的是 2017 年的序列建模与并行训练问题，不等于今天超长上下文与推理系统里的最优 attention 实现。
- 论文没有直接解决长上下文成本问题；后续很多工作实际上是在保留其建模接口的前提下，分别牺牲精确性、连接模式或硬件通用性来换效率。
- 当前 summary 聚焦其作为 attention 主线起点的地位，未细拆机器翻译实验与位置编码后续分支。

## 关联页面

- 主题：[注意力机制 Attention](../../wiki/topics/注意力机制%20Attention.md)
- 主题：[LLM 基础脉络](../../wiki/topics/LLM%20基础脉络.md)
- 概念：[Transformer](../../wiki/concepts/Transformer.md)
