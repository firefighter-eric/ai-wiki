---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Yang et al. - 2021 - Robust Transformer Modeling for Table-Text Encoding

## TL;DR（快速导读）

这篇表格文本模型研究减少行列顺序带来的虚假偏差，让表示更稳健地利用表格结构与文字关系。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

简单线性化表格会把序列位置混入语义。论文针对结构建模和表格文本对齐进行设计，并关注行列顺序扰动。应区分无关排序变化与真正改变内容对应关系的变化。

## 具体怎么理解

如果交换两行但保留各行数据，某些查询答案不应改变；模型不该仅凭原来的行位置作答。

## 关键事实

- **C1**：表格文本 TableFormer 以结构偏置减轻行列顺序扰动影响。
- **C2**：严格顺序不变不能回答依赖绝对行序的问题，并增加训练成本。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Yang%20et%20al.%20-%202021%20-%20Robust%20Transformer%20Modeling%20for%20Table-Text%20Encoding.pdf)
- 全文文本：[打开全文文本](../../raw/text/Yang%20et%20al.%20-%202021%20-%20Robust%20Transformer%20Modeling%20for%20Table-Text%20Encoding.md)
- 作者：Yang et al.
- 年份：2021
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Yang%20et%20al.%20-%202021%20-%20Robust%20Transformer%20Modeling%20for%20Table-Text%20Encoding.html)

## 争议与不确定点

- 长表成本仍受限制。
- 同名模型必须根据任务和来源区分。

## 关联页面

- 主题：[LLM RL](../topics/LLM%20RL.md)
- 综合：暂无
- [文档与表格的输入输出接口](../comparisons/%E6%96%87%E6%A1%A3%E4%B8%8E%E8%A1%A8%E6%A0%BC%E7%9A%84%E8%BE%93%E5%85%A5%E8%BE%93%E5%87%BA%E6%8E%A5%E5%8F%A3.md)：把本篇方法放到相关任务与比较条件中阅读。

## 方法与实验解读

该模型把表格结构放进注意力，避免无意义排列改变答案。但某些问题确实依赖第一行或排序，这时完全不变的表示反而不合适；需要先定义任务中的顺序是否有意义。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Yang%20et%20al.%20-%202021%20-%20Robust%20Transformer%20Modeling%20for%20Table-Text%20Encoding.md#source-section-2 ) | 这里不是图像表格恢复的同名 TableFormer |
| C2 | [原文]( ../../raw/text/Yang%20et%20al.%20-%202021%20-%20Robust%20Transformer%20Modeling%20for%20Table-Text%20Encoding.md#source-section-31 ) | 结构不变性有能力边界 |

## 核证范围

核对结构编码目标、行列增强实验和 §5.7 的绝对顺序局限。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
