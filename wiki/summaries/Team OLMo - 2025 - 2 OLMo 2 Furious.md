---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# OLMo Team - 2025 - 2 OLMo 2 Furious

## TL;DR（快速导读）

这份 OLMo 2 报告沿用 AdamW，并把数值稳定项和权重衰减范围作为训练稳定性的实验对象。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

可下载权重与可检查训练数据、代码和过程，是不同层次的开放；复现仍需匹配资源和设置。

## 来源信息

- 类型：技术报告 / arXiv 论文
- arXiv：https://arxiv.org/abs/2501.00656
- 原始 PDF：../../raw/pdf/Team OLMo - 2025 - 2 OLMo 2 Furious.pdf
- 发布页快照：../../raw/html/Team OLMo - 2025 - 2 OLMo 2 Furious.html
- 全文文本：../../raw/text/Team OLMo - 2025 - 2 OLMo 2 Furious.md
- 作者：OLMo Team
- 年份：2025
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

报告比较优化器 epsilon 设置，并讨论是否对词向量施加衰减。它说明优化器名称相同也可能对应不同训练行为；相关结论要与模型、数据和数值精度条件一起阅读。

## 关键事实

- **C1**：OLMo2将AdamW epsilon从1e-5降到1e-8，观察早期更新/梯度稳定改善。
- **C2**：weightdecay0.1但排除embedding，以避免小范数放大早期梯度。
- **C3**：报告同时研究初始化、QKnorm、zloss和数据配方。

## 争议与不确定点

- zloss的forward等价不保证不同kernel backward完全相同。
- 小规模趋势向大规模迁移需带width/lr与数据条件。

## 关联页面

- 概念：[OLMo 2](../concepts/OLMo%202.md)
- 概念：[Muon](../concepts/Muon.md)
- 对比：[Muon 与 AdamW](../comparisons/Muon%20与%20AdamW.md)
- 主题：[LLM 预训练](../topics/LLM%20预训练.md)

## 这里的术语是什么意思

- **embedding**：向量表示：把文字、图片等编码成一组数，用于模型计算或相似度比较。
- **weight decay**：权重衰减：训练中使权重逐步缩小的机制，需看它如何与梯度更新结合。

## 方法与实验解读

优化器参数通过更新尺度、embedding范数和normalization的Jacobian改变训练动态。报告提供可检查消融，说明AdamW仍是一套配方，而非一枚足以解释跨模型效果的标签。稳定性与下游质量也要分别确认。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Team%20OLMo%20-%202025%20-%202%20OLMo%202%20Furious.md#source-section-27 ) | 稳定性消融，不是任意模型epsilon越小越好。 |
| C2 | [原文]( ../../raw/text/Team%20OLMo%20-%202025%20-%202%20OLMo%202%20Furious.md#source-section-28 ) | parameter-group规则和系数一起看。 |
| C3 | [原文]( ../../raw/text/Team%20OLMo%20-%202025%20-%202%20OLMo%202%20Furious.md#source-section-15 ) | 整体改善不能只归因优化器名字。 |

## 核证范围

核读pretraining稳定性、epsilon与embedding decay消融及相关配方。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
