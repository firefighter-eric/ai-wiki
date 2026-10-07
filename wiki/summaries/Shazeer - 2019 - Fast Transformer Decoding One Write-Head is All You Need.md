---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Shazeer - 2019 - Fast Transformer Decoding One Write-Head is All You Need

## TL;DR（快速导读）

MQA 让多个查询头共享同一组键和值，减少逐词元生成时反复读取的缓存。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

理解一个词时，可对上下文不同位置分配权重；哪些位置可见由任务和注意力遮挡决定。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Shazeer - 2019 - Fast Transformer Decoding One Write-Head is All You Need.pdf
- 原始 HTML：../../raw/html/Shazeer - 2019 - Fast Transformer Decoding One Write-Head is All You Need.html
- 全文文本：../../raw/text/Shazeer - 2019 - Fast Transformer Decoding One Write-Head is All You Need.md
- 作者：Shazeer
- 年份：2019
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

它主要针对增量解码的内存带宽，而非让训练中的全序列注意力自动变成线性。共享降低缓存成本，也改变表示结构；质量、训练方式与真实解码速度都需检查。

## 关键事实

- **C1**：MQA多queryheads共享一组K/V，减少增量解码读取KV的带宽。
- **C2**：理论memory/computation比例中的长序列项减少head倍。
- **C3**：WMT14英德实验使用211M模型/6层、TPUv3和100ksteps。
- **C4**：速度评测为1024sequences、128input/128output的greedy增量推理。

## 争议与不确定点

- 质量代价依任务，原报告不是现代超大LLM普遍无损证明。
- prefill与decode瓶颈不同，收益需分别测。

## 关联页面

- 主题：[注意力机制 Attention](../../wiki/topics/注意力机制%20Attention.md)
- 概念：[Grouped-Query Attention](../../wiki/concepts/Grouped-Query%20Attention.md)
- 概念：[Transformer](../../wiki/concepts/Transformer.md)

## 这里的术语是什么意思

- **encoder**：编码器：把输入转成模型内部表示。

## 方法与实验解读

MQA针对自回归每步重复读取历史KV的成本，保留多query查询方式，却共享被查询的键和值。与稀疏attention减少访问位置不同；GQA进一步在多组KV和单组之间折中。不能把少KV存储自动当低总延迟。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Shazeer%20-%202019%20-%20Fast%20Transformer%20Decoding%20One%20Write-Head%20is%20All%20You%20Need.md#source-section-11 ) | 需训练适配，非任意MHA直接替换无损。 |
| C2 | [原文]( ../../raw/text/Shazeer%20-%202019%20-%20Fast%20Transformer%20Decoding%20One%20Write-Head%20is%20All%20You%20Need.md#source-section-12 ) | 理论项，不是全部walltime同倍加速。 |
| C3 | [原文]( ../../raw/text/Shazeer%20-%202019%20-%20Fast%20Transformer%20Decoding%20One%20Write-Head%20is%20All%20You%20Need.md#source-section-14 ) | 小型encoder-decoder实验。 |
| C4 | [原文]( ../../raw/text/Shazeer%20-%202019%20-%20Fast%20Transformer%20Decoding%20One%20Write-Head%20is%20All%20You%20Need.md#source-section-16 ) | batch、padding和硬件限定。 |

## 核证范围

核读MHA/MQA公式、复杂度、质量与增量速度实验。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
