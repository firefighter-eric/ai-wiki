---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Pang et al. - 2022 - Long Document Summarization with Top-down and Bottom-up Inference

## TL;DR（快速导读）

这篇长文摘要方法把局部到全局和全局到局部的信息推断结合起来，试图在有限成本下保留长文关键内容。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

仅由底层词语向上聚合，可能难以利用整篇文档的结构；完整自注意力又有较高成本。论文研究自上而下与自下而上的结合方式。需要检查模型实际覆盖的长度、摘要质量与事实一致性。

## 具体怎么理解

理解一章内容既要读具体句子，也要知道整章主旨；主旨又会帮助判断哪些句子重要。

## 关键事实

- **C1**：bottom-up 用局部注意力编码 token，top-down 引入高层长程信息修正表示。
- **C2**：评测包括 arXiv、PubMed 与 BookSum 等不同长度文档。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Pang%20et%20al.%20-%202022%20-%20Long%20Document%20Summarization%20with%20Top-down%20and%20Bottom-up%20Inference.pdf)
- 全文文本：[打开全文文本](../../raw/text/Pang%20et%20al.%20-%202022%20-%20Long%20Document%20Summarization%20with%20Top-down%20and%20Bottom-up%20Inference.md)
- 作者：Pang et al.
- 年份：2022
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Pang%20et%20al.%20-%202022%20-%20Long%20Document%20Summarization%20with%20Top-down%20and%20Bottom-up%20Inference.html)

## 争议与不确定点

- 摘要重叠指标不是事实保真证明。
- 局部窗口和层级结构会限制可传递信息。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无

## 方法与实验解读

该方法用层级结构兼顾局部细节与整篇主题，再由解码器生成摘要。它改善长文表示，但长文压缩仍可能丢掉关键事实，必须另检查事实覆盖和逻辑衔接。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Pang%20et%20al.%20-%202022%20-%20Long%20Document%20Summarization%20with%20Top-down%20and%20Bottom-up%20Inference.md#source-section-5 ) | 局部与全局信息分工，而非每层全部全局注意力 |
| C2 | [原文]( ../../raw/text/Pang%20et%20al.%20-%202022%20-%20Long%20Document%20Summarization%20with%20Top-down%20and%20Bottom-up%20Inference.md#source-section-8 ) | 学术与叙事摘要的难点不同 |

## 核证范围

核对 §2.1–2.2 的双向层级推断与 §3 的数据范围。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
