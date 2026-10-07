---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Team, Google - 2024 - Gemma Open Models Based on Gemini Research and Technology

## TL;DR（快速导读）

初代 Gemma 报告介绍从 Google 研究经验发展出的较小开放模型，适合建立家族的训练与使用背景。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

先看家族脉络，再进入 Gemma 3、Gemma 4 或 DiffusionGemma；后者的文本生成方式与普通自回归模型不同。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Team, Google - 2024 - Gemma Open Models Based on Gemini Research and Technology.pdf
- 全文文本：../../raw/text/Team, Google - 2024 - Gemma Open Models Based on Gemini Research and Technology.md
- 作者：Team, Google
- 年份：2024
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

报告涉及模型、训练和安全框架。阅读时需区分能力来源、具体规格与发布条件；后续版本的上下文、多模态或代理功能，不能直接回填到这一代模型。

## 关键事实

- **C1**：初代Gemma提供2B/7B，训练上下文8192。
- **C2**：2B使用MQA，7B用MHA；采用RoPE等结构。
- **C3**：2B/7B分别训练2T/6Ttokens，以English为主，明确非多模态。
- **C4**：instruction使用SFT和REINFORCE变体RLHF，而非本文PPO。

## 争议与不确定点

- 部署可及性不等同低资源机器都可无损运行。
- 安全基准与现实鲁棒性仍有距离。

## 关联页面

- 概念：[Gemma](../../wiki/concepts/Gemma.md)
- 概念：[Gemma 2](../../wiki/concepts/Gemma%202.md)
- 概念：[Gemma 3](../../wiki/concepts/Gemma%203.md)
- 主题：[LLM 预训练](../../wiki/topics/LLM%20预训练.md)

## 这里的术语是什么意思

- **benchmark**：评测基准：特定数据、任务与规则的组合，分数只在这些条件下成立。

## 方法与实验解读

小尺寸、特定attention和训练预算组成开放家族起点。模型虽承接Gemini技术，但报告明确收窄语言和模态目标；后续家族分析必须保留代际区别。后训练塑形不能替代外部知识证据。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Team%2C%20Google%20-%202024%20-%20Gemma%20Open%20Models%20Based%20on%20Gemini%20Research%20and%20Technology.md#source-section-4 ) | 不是后来Gemma3/4的多模态型号。 |
| C2 | [原文]( ../../raw/text/Team%2C%20Google%20-%202024%20-%20Gemma%20Open%20Models%20Based%20on%20Gemini%20Research%20and%20Technology.md#source-section-4 ) | 按尺寸区分attention。 |
| C3 | [原文]( ../../raw/text/Team%2C%20Google%20-%202024%20-%20Gemma%20Open%20Models%20Based%20on%20Gemini%20Research%20and%20Technology.md#source-section-8 ) | 与Gemini研究关联不等同相同能力。 |
| C4 | [原文]( ../../raw/text/Team%2C%20Google%20-%202024%20-%20Gemma%20Open%20Models%20Based%20on%20Gemini%20Research%20and%20Technology.md#source-section-14 ) | 具体算法以披露为准。 |

## 核证范围

核读架构/型号、预训练来源、SFT/RLHF与评测限制。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
