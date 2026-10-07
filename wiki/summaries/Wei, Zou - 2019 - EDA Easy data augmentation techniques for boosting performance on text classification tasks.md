---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Wei, Zou - 2019 - EDA Easy data augmentation techniques for boosting performance on text classification tasks

## TL;DR（快速导读）

EDA 用同义替换、随机插入、交换和删除扩充文本分类数据，方法简单，适合研究小数据场景的增强。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

数据不足时，轻量扰动可以提供额外训练样本。四种操作并不总能保留原句含义，使用时应检查标签是否仍然成立。原文收益来自特定分类任务，不能无条件迁到推理或事实任务。

## 具体怎么理解

把否定词删掉可能把“我不喜欢”变成相反意思；增强样本更多，也可能引入错误标签。

## 关键事实

- **C1**：用同义替换、随机插入、交换、删除四种增强操作训练文本分类。
- **C2**：数据充足时增益较小，预训练模型下也可能无明显收益。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Wei%2C%20Zou%20-%202019%20-%20EDA%20Easy%20data%20augmentation%20techniques%20for%20boosting%20performance%20on%20text%20classification%20tasks.pdf)
- 全文文本：[打开全文文本](../../raw/text/Wei%2C%20Zou%20-%202019%20-%20EDA%20Easy%20data%20augmentation%20techniques%20for%20boosting%20performance%20on%20text%20classification%20tasks.md)
- 作者：Wei, Zou
- 年份：2019
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Wei%2C%20Zou%20-%202019%20-%20EDA%20Easy%20data%20augmentation%20techniques%20for%20boosting%20performance%20on%20text%20classification%20tasks.html)

## 争议与不确定点

- 实体、否定词或关系被改动可能破坏原标签。
- 平均收益不保证每个数据集收益。

## 关联页面

- 主题：[传统NLP](../topics/传统%20NLP.md)
- 综合：暂无

## 方法与实验解读

EDA 用简单扰动扩充训练样本，重点是降低小数据分类过拟合。过多删除或替换会改变句意与标签，因此增强强度要与任务验证一起选择。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Wei%2C%20Zou%20-%202019%20-%20EDA%20Easy%20data%20augmentation%20techniques%20for%20boosting%20performance%20on%20text%20classification%20tasks.md#source-section-2 ) | 假设增强后标签仍有效 |
| C2 | [原文]( ../../raw/text/Wei%2C%20Zou%20-%202019%20-%20EDA%20Easy%20data%20augmentation%20techniques%20for%20boosting%20performance%20on%20text%20classification%20tasks.md#source-section-15 ) | 不能把少数据 CNN / RNN 结果外推到所有 LLM |

## 核证范围

核对四类操作、§4.4 强度消融和 §6 局限。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
