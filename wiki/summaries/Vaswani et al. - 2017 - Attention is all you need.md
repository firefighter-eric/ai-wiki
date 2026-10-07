---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Vaswani et al. - 2017 - Attention is all you need

## TL;DR（快速导读）

Transformer 用注意力与位置编码处理序列，让各位置直接获取其他位置的信息，取代循环计算作为主要结构。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Vaswani et al. - 2017 - Attention is all you need.pdf
- 原始 HTML：../../raw/html/Vaswani et al. - 2017 - Attention is all you need.html
- 全文文本：../../raw/text/Vaswani et al. - 2017 - Attention is all you need.md
- 作者：Vaswani et al.
- 年份：2017
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

论文组合缩放点积、多头注意力与前馈层，建立标准序列模型。全连接自注意力的计算随长度较快增长，后续方法分别从连接、近似、缓存与硬件实现优化；这些路线改变的层次不同。

## 关键事实

- **C1**：scaled-dotproductattention为softmax(QKᵀ/√dk)V，multihead有独立投影再拼接。
- **C2**：encoder双向selfattention，decoder因果selfattention和encoder-decoderattention。
- **C3**：全位置交互的attention计算随长度二次增长，并缩短远依赖路径。

## 争议与不确定点

- 原论文翻译结论不能直接当所有现代LLM能力解释。
- 二次计算与二次显存不是所有优化实现必然同时出现。

## 关联页面

- 主题：[注意力机制 Attention](../../wiki/topics/注意力机制%20Attention.md)
- 主题：[LLM 预训练](../../wiki/topics/LLM%20预训练.md)
- 概念：[Transformer](../../wiki/concepts/Transformer.md)
- [Google Research](../authors/Google%20Research.md)：沿作者或机构继续阅读相关来源。

## 这里的术语是什么意思

- **IO**：数据读写：计算与存储之间搬运数据的成本，可能成为速度瓶颈。

## 方法与实验解读

attention按内容从各位置取信息，FFN逐位置变换，位置编码补顺序。Transformer替代循环的训练顺序瓶颈，但decoder生成仍依赖前文。后续稀疏/低秩改变交互，FlashAttention保持同一算子却改变物化方式，MQA/GQA改变KV头数，不能都称同一种线性优化。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Vaswani%20et%20al.%20-%202017%20-%20Attention%20is%20all%20you%20need.md#source-section-10 ) | 标准定义，head不是简单重复。 |
| C2 | [原文]( ../../raw/text/Vaswani%20et%20al.%20-%202017%20-%20Attention%20is%20all%20you%20need.md#source-section-12 ) | 训练时可并行与生成时仍自回归区分。 |
| C3 | [原文]( ../../raw/text/Vaswani%20et%20al.%20-%202017%20-%20Attention%20is%20all%20you%20need.md#source-section-16 ) | 经典显存物化与后续IO优化区别。 |

## 核证范围

核读算子、multihead、mask/交叉attention、复杂度与训练/翻译设置。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
