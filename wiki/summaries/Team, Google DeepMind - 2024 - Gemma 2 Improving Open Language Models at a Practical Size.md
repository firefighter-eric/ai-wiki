---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Team, Google DeepMind - 2024 - Gemma 2 Improving Open Language Models at a Practical Size

## TL;DR（快速导读）

Gemma 2 报告讨论实用模型规模下的能力与部署取舍，帮助辨认相对初代的改进。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

比较两代模型时，应保持任务和使用方式一致，并确认尺寸变化，不能只比较家族名称。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Team, Google DeepMind - 2024 - Gemma 2 Improving Open Language Models at a Practical Size.pdf
- 全文文本：../../raw/text/Team, Google DeepMind - 2024 - Gemma 2 Improving Open Language Models at a Practical Size.md
- 作者：Team, Google DeepMind
- 年份：2024
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

它是家族继续迭代的技术来源。比较时应选定具体尺寸，并控制训练、推理和评测条件；能力密度、设备可用性和长上下文支持是需要分别核对的指标。

## 关键事实

- **C1**：Gemma2交替local4096与global8192attention，并采用logitsoftcapping等改动。
- **C2**：蒸馏使用teacher的每token概率训练student，不只是采样文本SFT。
- **C3**：2B/500Btokens消融从scratch平均60.3到distilled67.7。

## 争议与不确定点

- 蒸馏收益依teacher/任务/预算，不能视为任意模型配方常数。
- 报告多项架构与后训练共同改变，单项因果要回消融。

## 关联页面

- 概念：[Gemma 2](../../wiki/concepts/Gemma%202.md)
- 概念：[Gemma](../../wiki/concepts/Gemma.md)
- 概念：[Gemma 3](../../wiki/concepts/Gemma%203.md)
- 主题：[LLM 预训练](../../wiki/topics/LLM%20预训练.md)

## 方法与实验解读

结构降低部分attention成本，蒸馏把teacher分布中的软监督传递给小模型。报告将算力最优token预算与实际过训练/蒸馏区分，说明小模型质量优势需要把teacher成本及部署收益分开核算。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Team%2C%20Google%20DeepMind%20-%202024%20-%20Gemma%202%20Improving%20Open%20Language%20Models%20at%20a%20Practical%20Size.md#source-section-4 ) | global仍保留长依赖。 |
| C2 | [原文]( ../../raw/text/Team%2C%20Google%20DeepMind%20-%202024%20-%20Gemma%202%20Improving%20Open%20Language%20Models%20at%20a%20Practical%20Size.md#source-section-7 ) | soft-target目标。 |
| C3 | [原文]( ../../raw/text/Team%2C%20Google%20DeepMind%20-%202024%20-%20Gemma%202%20Improving%20Open%20Language%20Models%20at%20a%20Practical%20Size.md#source-section-11 ) | 3个bench均值及7Bteacher条件。 |

## 核证范围

核读架构、soft-target蒸馏、2B消融、评测与讨论。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
