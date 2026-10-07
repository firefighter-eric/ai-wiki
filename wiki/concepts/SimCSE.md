---
type: concept
---
# SimCSE

## TL;DR（快速导读）

SimCSE 用对比学习适配句向量，无监督版本把同一句话的不同 dropout 表示视为正例，方法简洁。

## 简介

SimCSE 用对比学习适配句向量，无监督版本把同一句话的不同 dropout 表示视为正例，方法简洁。

## 具体怎么理解

训练把同义表示拉近，但不能把全部句子挤到同一位置；区分正例与其他样本是关键。

## 关键属性

- 类型：句向量学习方法
- 代表来源：[Gao, Yao, Chen - 2021 - SimCSE Simple Contrastive Learning of Sentence Embeddings](../../wiki/summaries/Gao,%20Yao,%20Chen%20-%202021%20-%20SimCSE%20Simple%20Contrastive%20Learning%20of%20Sentence%20Embeddings.md)
- 当前角色：句向量主线的重要代表

## 相关主张

- SimCSE 说明简单的对比学习设计即可显著提升句向量质量。
- 在当前知识库里，它与 Sentence-BERT、ConSERT、DeCLUTR 处于同一类能力谱系。

## 来源支持

- [Gao, Yao, Chen - 2021 - SimCSE Simple Contrastive Learning of Sentence Embeddings](../../wiki/summaries/Gao,%20Yao,%20Chen%20-%202021%20-%20SimCSE%20Simple%20Contrastive%20Learning%20of%20Sentence%20Embeddings.md)

## 关联页面

- [Sentence-BERT](./Sentence-BERT.md)
- [Dense Retrieval](./Dense Retrieval.md)
- [BERT类双向Transformer语言模型](../topics/BERT%E7%B1%BB%E5%8F%8C%E5%90%91Transformer%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)
- [传统 NLP](../topics/传统%20NLP.md)
