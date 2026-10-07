---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Chowdhery et al. - 2022 - PaLM Scaling Language Modeling with Pathways

## TL;DR（快速导读）

PaLM 用大规模密集 Transformer 和 Pathways 训练系统研究语言模型的规模化收益；它同时是一份模型与训练系统报告。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

报告说明模型如何在大型 TPU 系统上训练，并通过语言、推理等任务研究扩大规模的效果。模型规模、训练数据和系统效率应一起阅读；不能把规模增长直接写成每类任务都按同一倍率进步。

## 具体怎么理解

同样是增加训练投入，一部分用于更大模型，一部分用于更多数据；两种分配方式可能得到不同效果。

## 关键事实

- **C1**：PaLM 是 dense decoder-only Transformer，采用 SwiGLU 等设计；540B 模型通过 Pathways 在两个 TPU v4 Pod、共 6144 芯片上训练。
- **C2**：三种规模模型使用同样打乱的一遍 780B tokens 数据，包括网络、书籍、Wikipedia、新闻、代码与社交对话。
- **C3**：GSM8K 的 58% 结果结合 540B、8-shot CoT 与外部计算器，不能标成纯无工具 zero-shot 结果。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Chowdhery%20et%20al.%20-%202022%20-%20PaLM%20Scaling%20Language%20Modeling%20with%20Pathways.pdf)
- 全文文本：[打开全文文本](../../raw/text/Chowdhery%20et%20al.%20-%202022%20-%20PaLM%20Scaling%20Language%20Modeling%20with%20Pathways.md)
- 作者：Chowdhery et al.
- 年份：2022
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Chowdhery%20et%20al.%20-%202022%20-%20PaLM%20Scaling%20Language%20Modeling%20with%20Pathways.html)

## 争议与不确定点

- 生成代码可能带细微错误，流畅解释与可运行正确性不能等同。
- 报告有偏见、记忆与数据污染分析；规模增长不会自动消除这些问题。
- 不同规模固定数据量的结果不直接回答 Chinchilla 式固定计算预算问题。

## 关联页面

- 主题：[LLM预训练](../topics/LLM%20预训练.md)
- 综合：暂无

## 方法与实验解读

论文同时展示大规模训练系统与少样例任务能力。推理实验把解题过程作为示例，必要时调用计算器；因此最终分数是模型、提示和辅助计算共同作用的结果。阅读时分别记下预训练成本、推理示例数量、工具和评测集。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Chowdhery%20et%20al.%20-%202022%20-%20PaLM%20Scaling%20Language%20Modeling%20with%20Pathways.md#source-section-8 ) | 系统并行与模型能力是不同维度；没有使用该训练设置下的 pipeline parallelism。 |
| C2 | [原文]( ../../raw/text/Chowdhery%20et%20al.%20-%202022%20-%20PaLM%20Scaling%20Language%20Modeling%20with%20Pathways.md#source-section-7 ) | 同一数据规模的模型比较有助于研究规模效应，不代表每个规模的训练计算都最优。 |
| C3 | [原文]( ../../raw/text/Chowdhery%20et%20al.%20-%202022%20-%20PaLM%20Scaling%20Language%20Modeling%20with%20Pathways.md#source-section-18 ) | 按作者报告的提示与计算器条件；该结果不能直接与不同工具预算分数比较。 |

## 核证范围

核对 §2 架构、§3 数据、§4 基础设施、§5 训练、§6.3 推理与工具条件及代码风险讨论。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
