---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# DeepSeek AI - 2025 - DeepSeek-R1-0528 Release

## TL;DR（快速导读）

R1-0528 是 R1 的更新发布页，说明推理模型怎样继续改善交互、结构化输出和工具调用。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

例如，回答需要按固定 JSON 字段交给查询工具时，输出格式可用与查询结果正确是两项检查；发布页说明前者的接口支持，不代替后者的任务验证。

## 来源信息

- 类型：官方发布页 / 模型更新资料
- 原始 HTML：[raw/html/DeepSeek AI - 2025 - DeepSeek-R1-0528 Release.html](../../raw/html/DeepSeek%20AI%20-%202025%20-%20DeepSeek-R1-0528%20Release.html)
- 全文文本：[raw/text/DeepSeek AI - 2025 - DeepSeek-R1-0528 Release.md](../../raw/text/DeepSeek%20AI%20-%202025%20-%20DeepSeek-R1-0528%20Release.md)
- 来源 URL：https://api-docs.deepseek.com/news/news250528
- 作者：DeepSeek AI
- 年份：2025
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

官方页面报告能力与可靠性改进，并介绍 JSON 输出和函数调用支持。它适合核对该版本的使用变化；强化学习为何产生推理行为，仍应回到 R1 技术报告，发布页不能替代训练机制证据。

## 关键事实

- **C1**：发布页称 R1-0528 改善基准、前端能力，并减少幻觉。
- **C2**：该版本发布时支持 JSON output 与 function calling，并称 API 使用方式不变。
- **C3**：发布页链接开放权重和 thinking-mode 使用指南。

## 争议与不确定点

- 没有正文实验表或消融，不能用它单独支撑性能提升幅度。
- 发布时 API 一致不保证所有后来端点保持不变。

## 关联页面

- 主题：[DeepSeek 系列](../topics/DeepSeek%20系列.md)
- 主题：[LLM RL](../topics/LLM%20RL.md)
- 概念：[DeepSeek](../concepts/DeepSeek.md)
- 概念：[DeepSeek-R1](../concepts/DeepSeek-R1.md)
- 概念：[GRPO](../concepts/GRPO.md)
- [DeepSeek](../authors/DeepSeek.md)：沿作者或机构继续阅读相关来源。

## 这里的术语是什么意思

- **benchmark**：评测基准：特定数据、任务与规则的组合，分数只在这些条件下成立。

## 方法与实验解读

来源是简短 release note，应作为版本差异的定位入口。它可以证明作者发布了哪些能力声明，却不足以给出减少多少幻觉、何种工具任务成功率等数量结论。若要作模型比较，需补读链接中的模型卡/报告及具体任务。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/DeepSeek%20AI%20-%202025%20-%20DeepSeek-R1-0528%20Release.md#source-section-1 ) | 简短官方发布声明，正文未提供逐项实验条件。 |
| C2 | [原文]( ../../raw/text/DeepSeek%20AI%20-%202025%20-%20DeepSeek-R1-0528%20Release.md#source-section-1 ) | 接口能力与调用正确率不同。 |
| C3 | [原文]( ../../raw/text/DeepSeek%20AI%20-%202025%20-%20DeepSeek-R1-0528%20Release.md#source-section-1 ) | 本页未将外链全文自动视为已经读过。 |

## 核证范围

核读完整 release note，仅核证明确文字声明，不从配图猜测数字。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
