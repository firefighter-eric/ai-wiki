---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Nogueira, Cho - 2019 - Passage Re-ranking with BERT

## TL;DR（快速导读）

BERT 重排序将问题和候选片段一起输入模型，直接判断相关性，适合在初步召回之后精细筛选。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

“苹果发布手机”和“苹果很甜”中的同一词语，因上下文不同而应有不同表示。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Nogueira, Cho - 2019 - Passage Re-ranking with BERT.pdf
- 原始 HTML：../../raw/html/Nogueira, Cho - 2019 - Passage Re-ranking with BERT.html
- 全文文本：../../raw/text/Nogueira, Cho - 2019 - Passage Re-ranking with BERT.md
- 作者：Nogueira, Cho
- 年份：2019
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

联合输入允许问题词元与文档词元充分交互，但每个候选都要计算一次。它提高的是候选排序质量，不能找回召回阶段完全遗漏的证据；候选数量决定效果与延迟的取舍。

## 关键事实

- **C1**：BERT reranker将query/passsage拼接，用CLS进行相关性二分类，query最多64tokens、合并最多512。
- **C2**：训练在BM25 top1000候选上使用正负样本交叉熵。
- **C3**：100k query-passage对、少于MSMARCO训练数据0.3%时，报告比IR-NET高1.4 MRR@10点。

## 争议与不确定点

- 历史SOTA和27%相对改善必须带原baseline，不能改写成准确率27个百分点。
- 数据效率结论限定特定预训练/任务，不说明零样本即可完成领域检索。

## 关联页面

- 主题：[搜索排序](../../wiki/topics/搜索排序.md)
- 主题：[BERT类双向Transformer语言模型](../../wiki/topics/BERT类双向Transformer语言模型.md)
- 概念：[ColBERT](../../wiki/concepts/ColBERT.md)

## 这里的术语是什么意思

- **cross-encoder**：交叉编码器：把查询和候选一起输入模型，进行较细的交互比较。
- **reranker**：重排序模型：对初步召回的候选进一步排序，通常比召回阶段更贵。
- **encoder**：编码器：把输入转成模型内部表示。

## 方法与实验解读

每个候选与query共同编码，早期交互能捕捉细粒度匹配，但无法给全库每篇都经济地打分。两阶段方案将召回与精排成本分开；先测候选Recall，再测候选内MRR，才能定位漏检发生在哪一层。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Nogueira%2C%20Cho%20-%202019%20-%20Passage%20Re-ranking%20with%20BERT.md#source-section-6 ) | 模型为BERTLarge，2019实验。 |
| C2 | [原文]( ../../raw/text/Nogueira%2C%20Cho%20-%202019%20-%20Passage%20Re-ranking%20with%20BERT.md#source-section-6 ) | 初召回决定候选覆盖。 |
| C3 | [原文]( ../../raw/text/Nogueira%2C%20Cho%20-%202019%20-%20Passage%20Re-ranking%20with%20BERT.md#source-section-13 ) | 数据效率的具体任务比较。 |

## 核证范围

核读方法、截断、训练候选、MSMARCO/TREC-CAR结果与训练规模实验。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
