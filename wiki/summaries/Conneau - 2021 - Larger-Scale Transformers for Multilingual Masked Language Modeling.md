---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Conneau - 2021 - Larger-Scale Transformers for Multilingual Masked Language Modeling

## TL;DR（快速导读）

XLM-RXL 和 XLM-RXXL 研究扩大多语言遮挡语言模型的收益：增加容量能改善部分跨语言理解任务，但评测条件仍重要。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

论文在 XLM-R 路线上训练更大的多语言模型，比较跨语言自然语言推断等任务的表现。阅读时应关注哪些语言和任务受益、训练数据覆盖，以及扩大规模的成本，而不是只看模型大小。

## 具体怎么理解

例如在一种语言上训练分类器，再测试另一种语言；跨语言能力要通过这样的迁移任务观察。

## 关键事实

- **C1**：在 XLM-R 的多语种 MLM 学习方式上扩展到 3.5B 与 10.7B 参数，训练仍以单语文本为基础。
- **C2**：XNLI 分别报告跨语言零样本迁移与 translate-train-all，二者需要分开理解。
- **C3**：与 mT5 的比较存在数据量、更新步数等差异，作者提醒难以作完全控制的比较。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Conneau%20-%202021%20-%20Larger-Scale%20Transformers%20for%20Multilingual%20Masked%20Language%20Modeling.pdf)
- 全文文本：[打开全文文本](../../raw/text/Conneau%20-%202021%20-%20Larger-Scale%20Transformers%20for%20Multilingual%20Masked%20Language%20Modeling.md)
- 作者：Conneau
- 年份：2021
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Conneau%20-%202021%20-%20Larger-Scale%20Transformers%20for%20Multilingual%20Masked%20Language%20Modeling.html)

## 争议与不确定点

- 基准覆盖的语言与任务有限，不能推导所有低资源语言都会同比受益。
- 大模型收益同时增加计算与部署成本。

## 关联页面

- 主题：[传统NLP](../topics/传统%20NLP.md)
- 综合：暂无

## 方法与实验解读

该工作检验扩大多语言编码器容量能否缓解语言间竞争。阅读结果时先确定下游训练使用英语还是所有语言的翻译数据，再比较同一协议下的模型；预训练数据和计算预算是不可省略的条件。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Conneau%20-%202021%20-%20Larger-Scale%20Transformers%20for%20Multilingual%20Masked%20Language%20Modeling.md#source-section-5 ) | 不是使用逐句对齐翻译监督预训练 |
| C2 | [原文]( ../../raw/text/Conneau%20-%202021%20-%20Larger-Scale%20Transformers%20for%20Multilingual%20Masked%20Language%20Modeling.md#source-section-7 ) | 翻译训练数据条件不能算纯英语零样本 |
| C3 | [原文]( ../../raw/text/Conneau%20-%202021%20-%20Larger-Scale%20Transformers%20for%20Multilingual%20Masked%20Language%20Modeling.md#source-section-14 ) | 容量优势不能从非控制比较单独归因 |

## 核证范围

核对 §2.1–2.2 的模型和 XNLI 协议、§3 的结果与 mT5 比较讨论。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
