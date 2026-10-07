---
type: concept
---
# Sentence-BERT

## TL;DR（快速导读）

Sentence-BERT 把句子独立编码成向量，再比较相似度，支持预计算文档表示和高效语义检索。

## 简介

Sentence-BERT 把句子独立编码成向量，再比较相似度，支持预计算文档表示和高效语义检索。

## 具体怎么理解

一万个常见问题先转成向量，新问题只编码一次再搜索；速度更高，但细粒度比较可能仍需重排序。

## 关键属性

- 类型：句向量模型
- 代表来源：[Sentence-BERT：孪生编码器句向量（Reimers 与 Gurevych，2019）](../../wiki/summaries/Devlin,%20Liu%20-%202014%20-%20Sentence-BERT%20Sentence%20Embeddings%20using%20Siamese%20BERT-Networks.md)
- 当前角色：句向量与语义检索的基础节点

## 相关主张

- Sentence-BERT 通过双塔式编码提升相似度计算效率与可用性。
- 在当前知识库里，它是 Dense Retrieval 和后续对比学习句向量方法的前置页。

## 来源支持

- [Sentence-BERT：孪生编码器句向量（Reimers 与 Gurevych，2019）](../../wiki/summaries/Devlin,%20Liu%20-%202014%20-%20Sentence-BERT%20Sentence%20Embeddings%20using%20Siamese%20BERT-Networks.md)

## 关联页面

- [BERT](./BERT.md)
- [SimCSE](./SimCSE.md)
- [Dense Retrieval](./Dense Retrieval.md)
- [BERT类双向Transformer语言模型](../topics/BERT%E7%B1%BB%E5%8F%8C%E5%90%91Transformer%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)
- [传统 NLP](../topics/传统%20NLP.md)
