---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Tay et al. - 2020 - Efficient Transformers A Survey

## TL;DR（快速导读）

《Efficient Transformers》按不同计算与内存瓶颈整理高效 Transformer 方法，帮助分辨各类“更快注意力”到底改了什么。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

高效方法可能改变连接范围、近似矩阵或重组执行。综述提供方法地图，但效果取决于长度、硬件和任务。稀疏或近似方法也不能与精确执行优化直接按名称比较。

## 具体怎么理解

减少可访问的词语与保留全部词语但减少显存读写，都是效率优化，却有不同的质量边界。

## 关键事实

- **C1**：围绕注意力的二次计算和内存复杂度组织高效 Transformer 方法。
- **C2**：作者指出不同基准、模型规模、超参数与预训练会混淆性能归因。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Tay%20et%20al.%20-%202020%20-%20Efficient%20Transformers%20A%20Survey.pdf)
- 全文文本：[打开全文文本](../../raw/text/Tay%20et%20al.%20-%202020%20-%20Efficient%20Transformers%20A%20Survey.md)
- 作者：Tay et al.
- 年份：2020
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Tay%20et%20al.%20-%202020%20-%20Efficient%20Transformers%20A%20Survey.html)

## 争议与不确定点

- 历史 taxonomy 不代表后续所有方法。
- 理论节约与目标硬件速度不同。

## 关联页面

- 主题：[LLM RL](../topics/LLM%20RL.md)
- 综合：暂无

## 方法与实验解读

高效注意力有不同代价：局部模式限制连通，低秩或核近似改变表示，记忆机制增加状态。应根据输入长度与任务选择，再测准确性、内存和延迟，而非只比较复杂度符号。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Tay%20et%20al.%20-%202020%20-%20Efficient%20Transformers%20A%20Survey.md#source-section-86 ) | 不覆盖全部推理栈成本 |
| C2 | [原文]( ../../raw/text/Tay%20et%20al.%20-%202020%20-%20Efficient%20Transformers%20A%20Survey.md#source-section-82 ) | 不能直接拼接各论文最佳数字 |

## 核证范围

核对方法组织、§4.1 评测可比性和 §4.3 全栈范围边界。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
