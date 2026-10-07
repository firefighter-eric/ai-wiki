---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Zhang et al. - 2022 - Tip-Adapter Training-Free Adaption of CLIP for Few-Shot Classification

## TL;DR（快速导读）

Tip-Adapter 利用少量标注样本的特征缓存辅助 CLIP 分类，研究无需完整重新训练的少样本适配方式。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

CLIP 的文字匹配能做零样本分类，但少量任务样本仍能提供额外信息。方法把这些样本的图像特征与标签组织成缓存，用相似度辅助预测。需区分免训练方案与后续可训练变体。

## 具体怎么理解

给每类几张示例图后，新图既与类别文字比较，也与缓存的示例特征比较。

## 关键事实

- **C1**：从有标签少样本构建 key–value cache，再与 CLIP 知识结合，无需优化训练即可适配分类。
- **C2**：Tip-Adapter-F 进一步训练 cache keys，与原本不训练的版本不同。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Zhang%20et%20al.%20-%202022%20-%20Tip-Adapter%20Training-Free%20Adaption%20of%20CLIP%20for%20Few-Shot%20Classification.pdf)
- 全文文本：[打开全文文本](../../raw/text/Zhang%20et%20al.%20-%202022%20-%20Tip-Adapter%20Training-Free%20Adaption%20of%20CLIP%20for%20Few-Shot%20Classification.md)
- 作者：Zhang et al.
- 年份：2022
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Zhang%20et%20al.%20-%202022%20-%20Tip-Adapter%20Training-Free%20Adaption%20of%20CLIP%20for%20Few-Shot%20Classification.html)

## 争议与不确定点

- 分类任务的收益不直接证明开放式问答或生成能力。
- 是否使用验证集选择参数应纳入比较。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无

## 方法与实验解读

Tip-Adapter 把少量示例存成缓存，以图像特征相似度补充 CLIP 的文字类别分数。它降低适配成本，但缓存规模、融合系数和类别数据会影响推理，不能把它误作纯零样本方案。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Zhang%20et%20al.%20-%202022%20-%20Tip-Adapter%20Training-Free%20Adaption%20of%20CLIP%20for%20Few-Shot%20Classification.md#source-section-9 ) | training-free 不等于不需要标签或没有调参 |
| C2 | [原文]( ../../raw/text/Zhang%20et%20al.%20-%202022%20-%20Tip-Adapter%20Training-Free%20Adaption%20of%20CLIP%20for%20Few-Shot%20Classification.md#source-section-19 ) | 两个版本的效果和成本分别比较 |

## 核证范围

核对 §3.1 的缓存适配及 §4 的 training-free / fine-tuned 版本比较。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
