---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Touvron et al. - 2023 - LLaMA Open and Efficient Foundation Language Models

## TL;DR（快速导读）

LLaMA 通过更多公开数据和更长训练，让较小模型在给定推理预算下获得较强语言能力。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Touvron et al. - 2023 - LLaMA Open and Efficient Foundation Language Models.pdf
- 全文文本：../../raw/text/Touvron et al. - 2023 - LLaMA Open and Efficient Foundation Language Models.md
- 作者：Touvron et al.
- 年份：2023
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

论文将训练投入与日后使用成本分开考虑，并发布多个规模的基础模型。阅读时可关注数据、训练预算与任务结果；较小参数量降低部分推理成本，但不代表所有任务都能替代更大的模型。

## 关键事实

- **C1**：LLaMA提供7B/13B/33B/65B，报告使用公开可得数据。
- **C2**：作者考虑推理预算，主张较小模型多训练与训练compute-optimal是不同目标。
- **C3**：AdamW配置beta0.9/0.95、decay0.1、clip1与cosine。

## 争议与不确定点

- 原始研究许可与后来Llama2许可不能混用。
- 公开数据来源仍有污染、来源偏差与权利问题。

## 关联页面

- 概念：[Llama 家族](../../wiki/concepts/Llama%20家族.md)
- 概念：[LLaMA（初代）](../../wiki/concepts/LLaMA%20初代.md)
- 概念：[Llama 2](../../wiki/concepts/Llama%202.md)
- 主题：[LLM预训练](../../wiki/topics/LLM%20预训练.md)
- [Hugo Touvron](../authors/Hugo%20Touvron.md)：沿作者或机构继续阅读相关来源。
- [Meta AI](../authors/Meta%20AI.md)：沿作者或机构继续阅读相关来源。

## 这里的术语是什么意思

- **benchmark**：评测基准：特定数据、任务与规则的组合，分数只在这些条件下成立。

## 方法与实验解读

训练成本与长期服务成本分开核算，才能解释为何把小模型训得超过训练算力最优token数。报告在若干任务上小模型胜过旧大模型，证据支持特定配方/任务比较，不支持参数越小越好。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Touvron%20et%20al.%20-%202023%20-%20LLaMA%20Open%20and%20Efficient%20Foundation%20Language%20Models.md#source-section-2 ) | 公开可得不等同全部数据已开放或无许可限制。 |
| C2 | [原文]( ../../raw/text/Touvron%20et%20al.%20-%202023%20-%20LLaMA%20Open%20and%20Efficient%20Foundation%20Language%20Models.md#source-section-3 ) | 部署请求量参与折中。 |
| C3 | [原文]( ../../raw/text/Touvron%20et%20al.%20-%202023%20-%20LLaMA%20Open%20and%20Efficient%20Foundation%20Language%20Models.md#source-section-18 ) | 型号learningrate/batch另表。 |

## 核证范围

核读模型/来源、训练与服务预算动机、架构、optimizer与20任务评测。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
