---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Joshi et al. - 2020 - Spanbert Improving pre-training by representing and predicting spans

## TL;DR（快速导读）

SpanBERT 把连续文本片段一起遮住，再让边界表示预测被遮片段，强化对实体、答案跨度等连续内容的建模。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

BERT 常遮住分散词语，而许多任务关心整段短语。SpanBERT 改变遮挡单位，并训练片段边界承载内部信息。它与逐词预测的区别在于训练单位和表示目标，具体收益要结合问答、共指等任务观察。

## 具体怎么理解

例如把“纽约大学”整个遮住，而不是只遮“大学”；边界需要帮助恢复完整实体。

## 关键事实

- **C1**：SpanBERT 随机遮蔽连续完整词跨度，并以跨度两边的可见 token 表示预测跨度内部内容，增加 Span Boundary Objective。
- **C2**：采用最长 512 tokens 的单一连续序列，同时取消 NSP 与双片段采样。
- **C3**：实验在 BooksCorpus 与英语 Wikipedia 上重新实现 BERT-large 作对照，覆盖抽取 QA、共指消解、关系抽取和 GLUE；消融为节约资源使用 1.2M 步检查点。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Joshi%20et%20al.%20-%202020%20-%20Spanbert%20Improving%20pre-training%20by%20representing%20and%20predicting%20spans.pdf)
- 全文文本：[打开全文文本](../../raw/text/Joshi%20et%20al.%20-%202020%20-%20Spanbert%20Improving%20pre-training%20by%20representing%20and%20predicting%20spans.md)
- 作者：Joshi et al.
- 年份：2020
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Joshi%20et%20al.%20-%202020%20-%20Spanbert%20Improving%20pre-training%20by%20representing%20and%20predicting%20spans.html)

## 争议与不确定点

- 英语预训练与选定的跨度任务不能代表任意语言或生成任务。
- 消融预算与最终模型不同，逐项贡献需按对应表格读。
- 单序列与取消 NSP 同时变化，因果归因应保守。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无

## 方法与实验解读

如果遮住一句话中的整个人名，边界表示需要概括缺失的短语，才能把它补回来。这样的训练更贴近‘整段实体指的是什么’和‘答案跨哪些词’等任务。随机跨度、边界目标、单序列训练应分别看消融，避免把训练配方混成一个神奇模块。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Joshi%20et%20al.%20-%202020%20-%20Spanbert%20Improving%20pre-training%20by%20representing%20and%20predicting%20spans.md#source-section-10 ) | 预测不依赖被遮蔽跨度内部的 token 表示；区别于仅更换随机 mask 的 BERT。 |
| C2 | [原文]( ../../raw/text/Joshi%20et%20al.%20-%202020%20-%20Spanbert%20Improving%20pre-training%20by%20representing%20and%20predicting%20spans.md#source-section-11 ) | 这是联合改动，不能把全部增益归因于取消 NSP。 |
| C3 | [原文]( ../../raw/text/Joshi%20et%20al.%20-%202020%20-%20Spanbert%20Improving%20pre-training%20by%20representing%20and%20predicting%20spans.md#source-section-30 ) | 完整版结果与消融训练预算不同；跨度任务优势不等于所有任务同样受益。 |

## 核证范围

核对 §3 三项改动、§4.2 对照设置、§5 结果类别与 §6 消融条件。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
