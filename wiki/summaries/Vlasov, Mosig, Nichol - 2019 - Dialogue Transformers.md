---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Vlasov, Mosig, Nichol - 2019 - Dialogue Transformers

## TL;DR（快速导读）

Dialogue Transformers 用注意力读取历史对话轮次，为对话系统选择下一步行动，研究哪些历史信息真正相关。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

多轮对话的决策不能只看用户最后一句。模型在对话历史上使用自注意力，使不同轮次对当前行动产生不同影响。它研究对话策略，与直接生成自然语言回答的任务需要区分。

## 具体怎么理解

用户先说明订单号，再补充退款原因；系统应结合先前信息决定查订单或转人工，而不是每轮重新开始。

## 关键事实

- **C1**：TED 在对话轮序列上使用 self-attention，并将当前状态与候选系统动作映射到共享空间进行检索。
- **C2**：训练提高正确动作与状态的点积相似度，并降低负样本动作相似度。
- **C3**：MultiWOZ 中长距离依赖较少，TED 与 LSTM 在该基准上表现相近。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Vlasov%2C%20Mosig%2C%20Nichol%20-%202019%20-%20Dialogue%20Transformers.pdf)
- 全文文本：[打开全文文本](../../raw/text/Vlasov%2C%20Mosig%2C%20Nichol%20-%202019%20-%20Dialogue%20Transformers.md)
- 作者：Vlasov, Mosig, Nichol
- 年份：2019
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Vlasov%2C%20Mosig%2C%20Nichol%20-%202019%20-%20Dialogue%20Transformers.html)

## 争议与不确定点

- 未标注语料上的 Recall@k 可能被泛用回复影响，论文并未把它作为主要行动正确性证据。
- 动作检索结论不能直接推广到自由生成客服回复。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 综合：暂无

## 方法与实验解读

TED 用注意力找出当前动作真正相关的历史轮次，再在动作集合里选择最匹配的一项。它适合有状态和动作定义的任务对话；对话历史是否含长程依赖，决定了注意力机制的优势能否显现。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Vlasov%2C%20Mosig%2C%20Nichol%20-%202019%20-%20Dialogue%20Transformers.md#source-section-10 ) | 是动作选择策略，不是开放式文本生成模型 |
| C2 | [原文]( ../../raw/text/Vlasov%2C%20Mosig%2C%20Nichol%20-%202019%20-%20Dialogue%20Transformers.md#source-section-13 ) | 依赖可枚举动作和训练标签 |
| C3 | [原文]( ../../raw/text/Vlasov%2C%20Mosig%2C%20Nichol%20-%202019%20-%20Dialogue%20Transformers.md#source-section-16 ) | 不能用它断言 Transformer 在所有对话政策上都更强 |

## 核证范围

核对 §III 的动作嵌入设计、§IV 的评测需求和 MultiWOZ 对照。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
