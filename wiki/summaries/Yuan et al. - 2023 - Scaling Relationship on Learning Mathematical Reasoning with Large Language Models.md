---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Yuan et al. - 2023 - Scaling Relationship on Learning Mathematical Reasoning with Large Language Models

## TL;DR（快速导读）

这篇数学推理缩放研究比较预训练损失、监督数据和增广数据的影响，发现参数量本身不是充分的能力指标。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

数学表现由基础模型质量和后续数据共同决定。作者研究不同因素与监督推理结果的关系。结果需要保持训练与评测条件，不应把某个相关指标当成对所有任务的因果解释。

## 具体怎么理解

两个参数量相同的模型，预训练质量不同，数学微调后的结果仍可能差很多。

## 关键事实

- **C1**：研究预训练损失、监督数据和增强数据与数学推理成绩的经验关系，发现损失比单看参数数更有解释力。
- **C2**：GSM8K 评测区分 greedy 单次解码与其他采样口径，训练也有固定 epoch 和预算。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Yuan%20et%20al.%20-%202023%20-%20Scaling%20Relationship%20on%20Learning%20Mathematical%20Reasoning%20with%20Large%20Language%20Models.pdf)
- 全文文本：[打开全文文本](../../raw/text/Yuan%20et%20al.%20-%202023%20-%20Scaling%20Relationship%20on%20Learning%20Mathematical%20Reasoning%20with%20Large%20Language%20Models.md)
- 作者：Yuan et al.
- 年份：2023
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Yuan%20et%20al.%20-%202023%20-%20Scaling%20Relationship%20on%20Learning%20Mathematical%20Reasoning%20with%20Large%20Language%20Models.html)

## 争议与不确定点

- 增强样本的质量与多样性不同于简单增加数量。
- 数学基准关系不能直接推广到开放式任务。

## 关联页面

- 主题：[LLM预训练](../topics/LLM%20预训练.md)
- 综合：暂无

## 方法与实验解读

数学能力与模型大小不必一一对应，训练质量和任务数据同样重要。论文的 scaling 拟合帮助规划实验，但外推前需要检查数据来源、底座损失和生成预算是否仍一致。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Yuan%20et%20al.%20-%202023%20-%20Scaling%20Relationship%20on%20Learning%20Mathematical%20Reasoning%20with%20Large%20Language%20Models.md#source-section-2 ) | 所测家族与数据范围，非一般因果定律 |
| C2 | [原文]( ../../raw/text/Yuan%20et%20al.%20-%202023%20-%20Scaling%20Relationship%20on%20Learning%20Mathematical%20Reasoning%20with%20Large%20Language%20Models.md#source-section-23 ) | 不能把多次采样收益当成单次能力 |

## 核证范围

核对摘要的经验关系、§5 结论与附录 A.1 的 GSM8K 训练解码协议。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
