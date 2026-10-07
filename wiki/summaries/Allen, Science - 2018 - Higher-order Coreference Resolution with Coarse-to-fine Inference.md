---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Allen, Science - 2018 - Higher-order Coreference Resolution with Coarse-to-fine Inference

## TL;DR（快速导读）

这篇共指消解方法反复利用可能的前文指代更新文本片段表示，并先粗筛候选，控制推理成本。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

共指消解要判断不同表达是否指向同一个对象。论文用前件概率分布作为注意力，对片段表示进行多轮更新，使模型能考虑已经形成的指代关系；粗到细的筛选避免对所有候选都做昂贵计算。

## 具体怎么理解

在“小王进门。他放下书包”中，“他”应连到“小王”；更长文章里，多次提及之间的关系还会互相影响。

## 关键事实

- **C1**：用先行词分布作为注意力，迭代修正 span 表示以考虑更高阶共指关系。
- **C2**：coarse-to-fine 先用便宜的打分剪枝，再做更精细的先行词比较。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Allen%2C%20Science%20-%202018%20-%20Higher-order%20Coreference%20Resolution%20with%20Coarse-to-fine%20Inference.pdf)
- 全文文本：[打开全文文本](../../raw/text/Allen%2C%20Science%20-%202018%20-%20Higher-order%20Coreference%20Resolution%20with%20Coarse-to-fine%20Inference.md)
- 作者：Allen, Science
- 年份：2018
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Allen%2C%20Science%20-%202018%20-%20Higher-order%20Coreference%20Resolution%20with%20Coarse-to-fine%20Inference.html)

## 争议与不确定点

- CoNLL-2012 英语数据上的增益不证明任意长文或语言效果。
- 候选漏掉真实先行词后，后续精排无法恢复它。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无

## 方法与实验解读

共指关系不能只看两处文本是否相似，还要考虑整组提及是否一致。该方法用迭代表示补充成对打分，同时通过粗筛控制长文成本；对知识抽取而言，错误共指可能把不同实体事实混在一起。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Allen%2C%20Science%20-%202018%20-%20Higher-order%20Coreference%20Resolution%20with%20Coarse-to-fine%20Inference.md#source-section-7 ) | 软近似推断，不是保证全局一致的精确求解 |
| C2 | [原文]( ../../raw/text/Allen%2C%20Science%20-%202018%20-%20Higher-order%20Coreference%20Resolution%20with%20Coarse-to-fine%20Inference.md#source-section-8 ) | 效率依赖候选保留和剪枝质量 |

## 核证范围

核对 §3 高阶近似、§4 粗细推断和 §5 英语实验设置。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
