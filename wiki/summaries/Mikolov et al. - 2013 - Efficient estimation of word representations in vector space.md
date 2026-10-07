---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Mikolov et al. - 2013 - Efficient estimation of word representations in vector space

## TL;DR（快速导读）

word2vec 用高效的词预测任务学习词向量，使词语能在连续空间中比较，成为后续语义表示方法的重要基础。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

论文提出用于大规模语料的表示学习结构，以较低计算成本学习词之间的关系。向量捕捉的是语料中的分布规律，并不能直接保证事实正确或理解完整句子。上下文歧义也要由后续模型处理。

## 具体怎么理解

例如常在相似语境出现的词可能向量接近；“苹果”的公司与水果含义在固定词向量中仍可能混在一起。

## 关键事实

- **C1**：CBOW 用上下文词的平均表示预测中心词，忽略上下文顺序。
- **C2**：Skip-gram 从中心词预测附近词，距离较远词的抽样权重较小。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Mikolov%20et%20al.%20-%202013%20-%20Efficient%20estimation%20of%20word%20representations%20in%20vector%20space.pdf)
- 全文文本：[打开全文文本](../../raw/text/Mikolov%20et%20al.%20-%202013%20-%20Efficient%20estimation%20of%20word%20representations%20in%20vector%20space.md)
- 作者：Mikolov et al.
- 年份：2013
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Mikolov%20et%20al.%20-%202013%20-%20Efficient%20estimation%20of%20word%20representations%20in%20vector%20space.html)

## 争议与不确定点

- 词类比结果不代表全部语义关系。
- 语料偏差也会进入词表示。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无

## 方法与实验解读

这篇 word2vec 工作展示了简单预测目标可以学到可用词向量。向量类比反映语料中的关系模式，但同一个词在不同语境仍共用向量，不能直接替代实体消歧或句级事实理解。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Mikolov%20et%20al.%20-%202013%20-%20Efficient%20estimation%20of%20word%20representations%20in%20vector%20space.md#source-section-11 ) | 静态词表示，不是上下文化句编码器 |
| C2 | [原文]( ../../raw/text/Mikolov%20et%20al.%20-%202013%20-%20Efficient%20estimation%20of%20word%20representations%20in%20vector%20space.md#source-section-12 ) | 窗口大小在质量与计算之间取舍 |

## 核证范围

核对 §3.1–3.2 的 CBOW / Skip-gram 目标及实验任务性质。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
