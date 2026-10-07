---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Yao et al. - 2021 - NLP From Scratch Without Large-Scale Pretraining A Simple and Efficient Framework

## TL;DR（快速导读）

TLM 用任务数据去大语料中找相关子集，再从头联合训练任务目标与语言目标，研究替代昂贵通用预训练的路径。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

通用预训练成本高，并非所有任务都需要整个语料。该框架围绕已知任务选择语料并训练。它与通用底座的差异在于任务依赖更强，不能直接假设同样具备广泛任务迁移能力。

## 具体怎么理解

已有分类样本可以帮助检索相似文章；这样训练更聚焦，但换成另一类任务时可能需要重新选择材料。

## 关键事实

- **C1**：TLM 先检索任务相关通用语料，再联合任务数据进行语言模型与任务学习。
- **C2**：八个任务、四个领域的实验比较不同预算；BM25 检索相关数据优于随机检索。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Yao%20et%20al.%20-%202021%20-%20NLP%20From%20Scratch%20Without%20Large-Scale%20Pretraining%20A%20Simple%20and%20Efficient%20Framework.pdf)
- 全文文本：[打开全文文本](../../raw/text/Yao%20et%20al.%20-%202021%20-%20NLP%20From%20Scratch%20Without%20Large-Scale%20Pretraining%20A%20Simple%20and%20Efficient%20Framework.md)
- 作者：Yao et al.
- 年份：2021
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Yao%20et%20al.%20-%202021%20-%20NLP%20From%20Scratch%20Without%20Large-Scale%20Pretraining%20A%20Simple%20and%20Efficient%20Framework.html)

## 争议与不确定点

- 任务专用模型未评估全部通用能力。
- 计算节约的比例绑定论文中的基线与预算，不能当成任何任务的固定节约率。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 综合：暂无

## 方法与实验解读

这篇论文提出任务明确时可以把训练资源集中在相关文本上。所谓 from scratch 并非不用外部数据，而是不先训练通用大模型。结果更适合解释有限任务的成本取舍，不能证明通用预训练失去价值。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Yao%20et%20al.%20-%202021%20-%20NLP%20From%20Scratch%20Without%20Large-Scale%20Pretraining%20A%20Simple%20and%20Efficient%20Framework.md#source-section-11 ) | 免除大规模任务无关预训练，仍有语言模型训练 |
| C2 | [原文]( ../../raw/text/Yao%20et%20al.%20-%202021%20-%20NLP%20From%20Scratch%20Without%20Large-Scale%20Pretraining%20A%20Simple%20and%20Efficient%20Framework.md#source-section-27 ) | 收益依赖任务数据和检索语料可用性 |

## 核证范围

核对 §3.1、§4 的八任务设定与数据检索消融。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
