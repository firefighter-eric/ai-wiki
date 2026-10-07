---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Sciavolino et al. - 2021 - Simple Entity-Centric Questions Challenge Dense Retrievers

## TL;DR（快速导读）

EntityQuestions 发现一些看似简单、围绕具体实体的问题会难倒稠密检索器，提醒检索能力受训练分布影响。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

向量检索在常见基准上表现好，不保证擅长所有实体。论文构造实体丰富的问题，检查检索退化。结果说明应按问题类型评估召回，实体名称和精确词面仍可能重要。

## 具体怎么理解

“某位不常见人物出生在哪里”语义简单，却包含罕见名字；泛化检索器可能召回语义相近但对象错误的材料。

## 关键事实

- **C1**：EntityQuestions 从 Wikidata 的 24 类关系构造实体问句，研究稠密检索的实体泛化。
- **C2**：在该基准上，NQ 训练的 DPR 明显落后 BM25，论文采用 top-20 检索准确率。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Sciavolino%20et%20al.%20-%202021%20-%20Simple%20Entity-Centric%20Questions%20Challenge%20Dense%20Retrievers.pdf)
- 全文文本：[打开全文文本](../../raw/text/Sciavolino%20et%20al.%20-%202021%20-%20Simple%20Entity-Centric%20Questions%20Challenge%20Dense%20Retrievers.md)
- 作者：Sciavolino et al.
- 年份：2021
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Sciavolino%20et%20al.%20-%202021%20-%20Simple%20Entity-Centric%20Questions%20Challenge%20Dense%20Retrievers.html)

## 争议与不确定点

- 模板与实体覆盖影响难度。
- 检索成功与最终答案正确是两个步骤。

## 关联页面

- 主题：[传统NLP](../topics/传统%20NLP.md)
- 综合：暂无
- [句向量、稠密召回与交互排序](../comparisons/%E5%8F%A5%E5%90%91%E9%87%8F%E3%80%81%E7%A8%A0%E5%AF%86%E5%8F%AC%E5%9B%9E%E4%B8%8E%E4%BA%A4%E4%BA%92%E6%8E%92%E5%BA%8F.md)：把本篇方法放到相关任务与比较条件中阅读。

## 方法与实验解读

常见问答集成绩好并不保证能找到冷门实体。该研究支持保留关键词检索与实体别名召回，再按真实查询检验稠密方案，避免只因为 embedding 更新就放弃稀疏检索。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Sciavolino%20et%20al.%20-%202021%20-%20Simple%20Entity-Centric%20Questions%20Challenge%20Dense%20Retrievers.md#source-section-9 ) | 模板事实问句不代表所有自然查询 |
| C2 | [原文]( ../../raw/text/Sciavolino%20et%20al.%20-%202021%20-%20Simple%20Entity-Centric%20Questions%20Challenge%20Dense%20Retrievers.md#source-section-10 ) | 具体训练数据和基准，不能推广为所有稠密检索较差 |

## 核证范围

核对数据构造、DPR / BM25 的训练与 top-20 协议及结论。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
