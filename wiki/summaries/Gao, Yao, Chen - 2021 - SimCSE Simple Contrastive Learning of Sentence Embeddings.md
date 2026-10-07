---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Gao, Yao, Chen - 2021 - SimCSE Simple Contrastive Learning of Sentence Embeddings

## TL;DR（快速导读）

SimCSE 用对比学习得到句向量：无监督版本把同一句话的两次 dropout 表示拉近，有监督版本利用推断数据构造正负例。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

原始预训练模型的句向量未必适合直接比较语义。SimCSE 用很少的结构改动训练表示，并报告 dropout 对避免坍塌的重要性。需要区分无监督版本与使用标注数据的版本，比较时也要保持训练数据条件一致。

## 具体怎么理解

同一句话经过两次随机 dropout 得到略不同表示，训练让它们靠近，同时与其他句子的表示区分开。

## 关键事实

- **C1**：无监督 SimCSE 把同一句话编码两次，独立 dropout mask 生成两个视图，以对比目标拉近正例并利用批内负例。
- **C2**：监督版本引入 NLI 数据的正例关系；STS 测试仍不使用 STS 训练集，监督指额外 NLI 标签。
- **C3**：七项 STS 上，BERT-base 无监督与监督版本的平均 Spearman 相关分别报告为 76.25 与 81.57。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Gao%2C%20Yao%2C%20Chen%20-%202021%20-%20SimCSE%20Simple%20Contrastive%20Learning%20of%20Sentence%20Embeddings.pdf)
- 全文文本：[打开全文文本](../../raw/text/Gao%2C%20Yao%2C%20Chen%20-%202021%20-%20SimCSE%20Simple%20Contrastive%20Learning%20of%20Sentence%20Embeddings.md)
- 作者：Gao, Yao, Chen
- 年份：2021
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Gao%2C%20Yao%2C%20Chen%20-%202021%20-%20SimCSE%20Simple%20Contrastive%20Learning%20of%20Sentence%20Embeddings.html)

## 争议与不确定点

- STS 相关性提升不能直接证明企业检索的召回与答案可靠性。
- batch、温度、负例与 pooling 选择影响结果；无监督与监督设置应分开。
- 原文的对齐与均匀性分析解释表示结构，不等于语义信息完整保存。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无
- [句向量、稠密召回与交互排序](../comparisons/%E5%8F%A5%E5%90%91%E9%87%8F%E3%80%81%E7%A8%A0%E5%AF%86%E5%8F%AC%E5%9B%9E%E4%B8%8E%E4%BA%A4%E4%BA%92%E6%8E%92%E5%BA%8F.md)：把本篇方法放到相关任务与比较条件中阅读。

## 方法与实验解读

把‘这家店周日不开门’送入同一编码器两遍，dropout 让两次内部表示略有不同，但训练要求它们仍相近。这个目标能改善向量空间结构；实际检索还要验证查询与文档是否属于相同分布，并处理否定、数字与专业领域。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Gao%2C%20Yao%2C%20Chen%20-%202021%20-%20SimCSE%20Simple%20Contrastive%20Learning%20of%20Sentence%20Embeddings.md#source-section-7 ) | 句子文本相同，随机性来自编码器 dropout，不是复制同一个确定向量。 |
| C2 | [原文]( ../../raw/text/Gao%2C%20Yao%2C%20Chen%20-%202021%20-%20SimCSE%20Simple%20Contrastive%20Learning%20of%20Sentence%20Embeddings.md#source-section-10 ) | 无监督 STS 评测不等于监督 SimCSE 没用标签。 |
| C3 | [原文]( ../../raw/text/Gao%2C%20Yao%2C%20Chen%20-%202021%20-%20SimCSE%20Simple%20Contrastive%20Learning%20of%20Sentence%20Embeddings.md#source-section-18 ) | 这些是特定编码器与评测协议的句子相似度相关，不是问答正确率。 |

## 核证范围

核对 §3 dropout 视图、§4 监督数据、§6.1 评测协议、§6.2 数值与 pooling 分析。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
