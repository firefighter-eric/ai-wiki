---
type: concept
---
# DPR

## TL;DR（快速导读）

DPR 分别把问题和文本段落编码成向量，再用向量相似度快速召回候选，是开放域问答的检索组件。

## 简介

DPR 分别把问题和文本段落编码成向量，再用向量相似度快速召回候选，是开放域问答的检索组件。

## 具体怎么理解

文档可以提前编码；新问题到来后搜索近邻。找到候选只是第一步，答案仍要核对是否在材料中。

## 关键属性

- 类型：检索模型
- 代表来源：[Karpukhin et al. - 2020 - Dense passage retrieval for open-domain question answering](../../wiki/summaries/Karpukhin%20et%20al.%20-%202020%20-%20Dense%20passage%20retrieval%20for%20open-domain%20question%20answering.md)
- 当前角色：Dense Retrieval 的经典实现页

## 相关主张

- DPR 用 query encoder 与 passage encoder 建立稠密检索基础范式。
- 在当前知识库里，它是后续 RAG 讨论的重要前置能力块。

## 来源支持

- [Karpukhin et al. - 2020 - Dense passage retrieval for open-domain question answering](../../wiki/summaries/Karpukhin%20et%20al.%20-%202020%20-%20Dense%20passage%20retrieval%20for%20open-domain%20question%20answering.md)

## 关联页面

- [Dense Retrieval](./Dense Retrieval.md)
- [Sentence-BERT](./Sentence-BERT.md)
- [传统 NLP](../topics/传统%20NLP.md)

## 这里的术语是什么意思

- **RAG**：检索增强生成：先找外部材料，再利用这些材料生成回答。
- **encoder**：编码器：把输入转成模型内部表示。
