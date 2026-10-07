---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Lozhkov et al. - 2024 - StarCoder 2 and The Stack v2 The Next Generation

## TL;DR（快速导读）

StarCoder2 与 The Stack v2 报告介绍代码模型及其训练数据，重点关注数据质量、授权处理与不同模型规模。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Lozhkov et al. - 2024 - StarCoder 2 and The Stack v2 The Next Generation.pdf
- 全文文本：../../raw/text/Lozhkov et al. - 2024 - StarCoder 2 and The Stack v2 The Next Generation.md
- 作者：Lozhkov et al.
- 年份：2024
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

这条路线把可复用代码语料和专门代码模型一起建设。使用时需核对代码任务、数据来源与发布许可；模型开放不等于所有训练内容具有相同授权，也不等于生成代码无需检查。

## 关键事实

- **C1**：StarCoder2含3B/7B/15B，报告各训练3.3–4.3T tokens；数据与TheStackv2协同建设。
- **C2**：SWH来源涵盖619语言，另加入PR、notebook和文档。
- **C3**：结构改用RoPE与GQA，保持较少KV heads。
- **C4**：7B在部分代码任务仍落后DeepSeekCoder6.7B。

## 争议与不确定点

- 代码数据存在授权、敏感内容与污染问题，报告专章讨论。
- 开放数据链和模型权重并不保证每个下游用途自动合适。

## 关联页面

- 概念：[StarCoder2](../../wiki/concepts/StarCoder2.md)
- 概念：[Code Llama](../../wiki/concepts/Code%20Llama.md)
- 主题：[LLM 预训练](../../wiki/topics/LLM%20预训练.md)
- 比较：[开放模型家族与中国重要家族对照](../../wiki/comparisons/开放模型家族与中国重要家族对照.md)

## 方法与实验解读

代码模型质量同时依赖来源覆盖、许可/去重流程和模型训练。Stack archive规模、筛后训练集和模型实际token数是三种分母。评测从生成到修复等任务分别衡量，不应把HumanEval成绩当生产软件工程成功率。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Lozhkov%20et%20al.%20-%202024%20-%20StarCoder%202%20and%20The%20Stack%20v2%20The%20Next%20Generation.md#source-section-2 ) | 训练子集与整个archive规模分开。 |
| C2 | [原文]( ../../raw/text/Lozhkov%20et%20al.%20-%202024%20-%20StarCoder%202%20and%20The%20Stack%20v2%20The%20Next%20Generation.md#source-section-2 ) | 语言计数不表示每语言同等训练/评测。 |
| C3 | [原文]( ../../raw/text/Lozhkov%20et%20al.%20-%202024%20-%20StarCoder%202%20and%20The%20Stack%20v2%20The%20Next%20Generation.md#source-section-76 ) | 相对StarCoderBase的版本变化。 |
| C4 | [原文]( ../../raw/text/Lozhkov%20et%20al.%20-%202024%20-%20StarCoder%202%20and%20The%20Stack%20v2%20The%20Next%20Generation.md#source-section-90 ) | 不把家族贡献写成所有尺寸全胜。 |

## 核证范围

核读数据/型号摘要、架构改动、7B结果与社会影响章节。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
