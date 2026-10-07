---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Sun et al. - 2023 - A Comparative Study between Full-Parameter and LoRA-based Fine-Tuning on Chinese Instruction Data for Instruction Fo

## TL;DR（快速导读）

这篇中文指令微调实验比较 LoRA 与全参数微调，关注节省训练成本之后，任务效果有怎样的变化。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

LoRA 只训练低秩增量，全参数微调更新全部权重。作者在中文指令数据上对照两种路线。需要一起看基础模型、数据量、训练预算与评测口径，不能只用参数量更少就判断方法更好。

## 具体怎么理解

当预算有限时，能训练的批量、时长和设置可能不同；公平比较应说明这些资源差异。

## 关键事实

- **C1**：使用不同规模中文指令数据及 LLaMA 底座比较 LoRA 与完整微调。
- **C2**：在该实验的 0.6M / 2M 数据比较中，完整微调成绩更好。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Sun%20et%20al.%20-%202023%20-%20A%20Comparative%20Study%20between%20Full-Parameter%20and%20LoRA-based%20Fine-Tuning%20on%20Chinese%20Instruction%20Data%20for%20Instruction%20Fo.pdf)
- 全文文本：[打开全文文本](../../raw/text/Sun%20et%20al.%20-%202023%20-%20A%20Comparative%20Study%20between%20Full-Parameter%20and%20LoRA-based%20Fine-Tuning%20on%20Chinese%20Instruction%20Data%20for%20Instruction%20Fo.md)
- 作者：Sun et al.
- 年份：2023
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Sun%20et%20al.%20-%202023%20-%20A%20Comparative%20Study%20between%20Full-Parameter%20and%20LoRA-based%20Fine-Tuning%20on%20Chinese%20Instruction%20Data%20for%20Instruction%20Fo.html)

## 争议与不确定点

- LoRA rank、层选择和训练预算限制结论外推。
- 评测结果不覆盖全部中文领域与指令形式。

## 关联页面

- 主题：[LLM预训练](../topics/LLM%20预训练.md)
- 综合：暂无

## 这里的术语是什么意思

- **LoRA**：低秩适配：冻结底座，只训练较小矩阵表示的权重增量。

## 方法与实验解读

这篇工作为 LoRA 的成本收益提供具体中文任务比较。应先对齐数据量与基座，再看任务表现和可训练参数；不能只因方法更轻就推断同等效果，也不能从一组实验否定所有低秩适配。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Sun%20et%20al.%20-%202023%20-%20A%20Comparative%20Study%20between%20Full-Parameter%20and%20LoRA-based%20Fine-Tuning%20on%20Chinese%20Instruction%20Data%20for%20Instruction%20Fo.md#source-section-8 ) | 训练数据规模、底座与任务同时影响结果 |
| C2 | [原文]( ../../raw/text/Sun%20et%20al.%20-%202023%20-%20A%20Comparative%20Study%20between%20Full-Parameter%20and%20LoRA-based%20Fine-Tuning%20on%20Chinese%20Instruction%20Data%20for%20Instruction%20Fo.md#source-section-12 ) | 局部实验结论，不是 LoRA 永远落后的定律 |

## 核证范围

核对 §4 数据规模、§4.4 对比与 §4.6 讨论。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
