---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Lu et al. - 2024 - Large Language Model for Table Processing A Survey

## TL;DR（快速导读）

这篇综述整理 LLM 处理表格的任务和方法，重点是二维结构怎样输入模型，以及模型怎样查询、理解和操作它。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

把表格简单写成一长串文字，可能丢掉行列、格式和关系。综述讨论表格问答、抽取与操作等任务，也涉及语言模型和视觉模型的不同输入路线。阅读时应先分清任务，不把所有表格问题都当作文本问答。

## 具体怎么理解

问“哪一行收入最高”和“把这张表恢复成电子表格”需要不同的输入、输出和评价方法。

## 关键事实

- **C1**：综述同时组织表格任务、基准与 LLM 方法，任务包含不同输入和输出要求。
- **C2**：指标包括执行准确率、exact match 与子问题准确率，不能直接跨任务比较。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Lu%20et%20al.%20-%202024%20-%20Large%20Language%20Model%20for%20Table%20Processing%20A%20Survey.pdf)
- 全文文本：[打开全文文本](../../raw/text/Lu%20et%20al.%20-%202024%20-%20Large%20Language%20Model%20for%20Table%20Processing%20A%20Survey.md)
- 作者：Lu et al.
- 年份：2024
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Lu%20et%20al.%20-%202024%20-%20Large%20Language%20Model%20for%20Table%20Processing%20A%20Survey.html)

## 争议与不确定点

- 训练型与提示型方案的成本、数据访问与错误机制不同。
- 汇总指标不能证明复杂分析结论正确。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 综合：暂无

## 方法与实验解读

表格领域要先判断系统拿到的是图片、结构化表还是数据库，再判断目标是读结构、回答问题还是执行操作。该综述提供任务地图；具体模型的事实仍需回到对应研究。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Lu%20et%20al.%20-%202024%20-%20Large%20Language%20Model%20for%20Table%20Processing%20A%20Survey.md#source-section-28 ) | 表格处理不是单一识别任务 |
| C2 | [原文]( ../../raw/text/Lu%20et%20al.%20-%202024%20-%20Large%20Language%20Model%20for%20Table%20Processing%20A%20Survey.md#source-section-12 ) | 执行正确与字符串相同是不同判据 |

## 核证范围

核对 §2.4 的基准与指标、结论中的任务与方法范围。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
