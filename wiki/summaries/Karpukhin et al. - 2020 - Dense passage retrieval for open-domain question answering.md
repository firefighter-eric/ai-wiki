---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Karpukhin et al. - 2020 - Dense passage retrieval for open-domain question answering

## TL;DR（快速导读）

DPR 分别把问题和文章片段编码成向量，便于提前建库，再按语义相似度找到回答证据。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Karpukhin et al. - 2020 - Dense passage retrieval for open-domain question answering.pdf
- 原始 HTML：../../raw/html/Karpukhin et al. - 2020 - Dense passage retrieval for open-domain question answering.html
- 全文文本：../../raw/text/Karpukhin et al. - 2020 - Dense passage retrieval for open-domain question answering.md
- 作者：Karpukhin et al.
- 年份：2020
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

双编码器使候选文档可以离线处理，查询时只计算问题表示并搜索向量。它解决的是召回，不直接保证答案正确；训练样本、负例和领域变化决定能否找回真正相关的片段。

## 关键事实

- **C1**：DPR用独立问题/段落编码器学习dense表示，以相似度召回passage。
- **C2**：训练需要positive及negative passage，使用正样本NLL与batch内负例。
- **C3**：作者报告top-20召回较BM25高9–19个百分点。
- **C4**：更高召回通常改善QA，但报告指出SQuAD等例外。

## 争议与不确定点

- 域迁移、负例与候选规模都可能改变结果。
- 高召回不能保证答案正确，也不解决来源权限与更新问题。

## 关联页面

- 概念：[Dense Retrieval](../../wiki/concepts/Dense%20Retrieval.md)
- 概念：[DPR](../../wiki/concepts/DPR.md)
- 主题：[AI 智能问答与智能客服](../../wiki/topics/AI%20%E6%99%BA%E8%83%BD%E9%97%AE%E7%AD%94%E4%B8%8E%E6%99%BA%E8%83%BD%E5%AE%A2%E6%9C%8D.md)
- [句向量、稠密召回与交互排序](../comparisons/%E5%8F%A5%E5%90%91%E9%87%8F%E3%80%81%E7%A8%A0%E5%AF%86%E5%8F%AC%E5%9B%9E%E4%B8%8E%E4%BA%A4%E4%BA%92%E6%8E%92%E5%BA%8F.md)：把本篇方法放到相关任务与比较条件中阅读。

## 这里的术语是什么意思

- **embedding**：向量表示：把文字、图片等编码成一组数，用于模型计算或相似度比较。

## 方法与实验解读

双编码器让文档向量可离线索引，在线只编码问题，速度适合大规模候选召回。监督正负对提供语义匹配方向，而reader负责在候选中找答案。企业库常有缩写、别称和精确编号，本库据此建议把dense与词面检索视为互补，再在本地问题集验证。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Karpukhin%20et%20al.%20-%202020%20-%20Dense%20passage%20retrieval%20for%20open-domain%20question%20answering.md#source-section-2 ) | 检索器与回答reader是两层。 |
| C2 | [原文]( ../../raw/text/Karpukhin%20et%20al.%20-%202020%20-%20Dense%20passage%20retrieval%20for%20open-domain%20question%20answering.md#source-section-9 ) | 负例选择影响检索区分度。 |
| C3 | [原文]( ../../raw/text/Karpukhin%20et%20al.%20-%202020%20-%20Dense%20passage%20retrieval%20for%20open-domain%20question%20answering.md#source-section-2 ) | 指定开放问答集合与BM25配置，不是所有领域规律。 |
| C4 | [原文]( ../../raw/text/Karpukhin%20et%20al.%20-%202020%20-%20Dense%20passage%20retrieval%20for%20open-domain%20question%20answering.md#source-section-28 ) | answer-containing recall与最终EM不同。 |

## 核证范围

核读dual-encoder、训练负例、数据集与检索/QA结果。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
