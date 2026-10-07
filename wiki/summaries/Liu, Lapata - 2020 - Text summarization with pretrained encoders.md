---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Liu, Lapata - 2020 - Text summarization with pretrained encoders

## TL;DR（快速导读）

这篇摘要工作使用预训练编码器建模文档，同时研究抽取式和生成式摘要，说明两类输出需要不同的训练与接口。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

抽取式摘要选择原句，生成式摘要重新组织文字。论文将 BERT 的预训练表示用于两类任务，并考虑文档级编码与生成训练的衔接。摘要分数不能代替事实一致性和可读性检查。

## 具体怎么理解

“挑出三句原话”和“用自己的话概括成一段”看似都叫摘要，实际输出机制与错误方式不同。

## 关键事实

- **C1**：BertSum 用文档级编码与句间层获得句子表示，抽取版对句子是否进入摘要做分类。
- **C2**：生成版用预训练 BertSum encoder 和随机初始化的六层 Transformer decoder，为缓解训练不匹配分别设置优化器。
- **C3**：评价覆盖 CNN/DailyMail、NYT 与 XSum；前者偏抽取，XSum 为高度概括的单句摘要，抽取方法在 XSum 表现较差。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Liu%2C%20Lapata%20-%202020%20-%20Text%20summarization%20with%20pretrained%20encoders.pdf)
- 全文文本：[打开全文文本](../../raw/text/Liu%2C%20Lapata%20-%202020%20-%20Text%20summarization%20with%20pretrained%20encoders.md)
- 作者：Liu, Lapata
- 年份：2020
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Liu%2C%20Lapata%20-%202020%20-%20Text%20summarization%20with%20pretrained%20encoders.html)

## 争议与不确定点

- 原 BERT 512 位置限制以新位置嵌入扩展，但这不直接保证任意长文能力。
- 新闻基准上的高 ROUGE 不证明综述事实忠实或证据完整。
- 数据集风格差异使‘抽取与生成谁更好’没有统一答案。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无

## 方法与实验解读

长报道可以摘出重要原句，也可以重新写成一句核心意思。BertSum 分别建模这两种任务，并研究先抽取后生成的训练路线。对于本库，抽取的句子适合回查来源，但 TL;DR 还需要连贯解释与事实核对，不能只优化摘要相似度分数。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Liu%2C%20Lapata%20-%202020%20-%20Text%20summarization%20with%20pretrained%20encoders.md#source-section-10 ) | 抽取摘要保留原句，不等于自由改写或保证信息完整。 |
| C2 | [原文]( ../../raw/text/Liu%2C%20Lapata%20-%202020%20-%20Text%20summarization%20with%20pretrained%20encoders.md#source-section-11 ) | 预训练编码器并没有直接成为成熟的文本生成器。 |
| C3 | [原文]( ../../raw/text/Liu%2C%20Lapata%20-%202020%20-%20Text%20summarization%20with%20pretrained%20encoders.md#source-section-21 ) | 数据的摘要风格决定方法排名；ROUGE 与人工评价需分别读。 |

## 核证范围

核对 §3.1–3.3 编码与两类摘要、§4.1 数据、§5.1 自动评价及人类评价路径。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
