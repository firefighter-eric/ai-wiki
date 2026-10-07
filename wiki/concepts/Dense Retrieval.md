---
type: concept
---
# Dense Retrieval

## TL;DR（快速导读）

稠密检索将查询与文档变成向量，按相似度寻找候选，能捕捉部分语义关系，但也会漏掉精确实体。

## 简介

稠密检索将查询与文档变成向量，按相似度寻找候选，能捕捉部分语义关系，但也会漏掉精确实体。

## 具体怎么理解

“怎么申请退款”可召回措辞不同的退款政策；遇到罕见产品名时，还应检查词面检索是否更可靠。

## 关键属性

- 类型：检索方法
- 代表来源：[Oğuz et al. - 2021 - Domain-matched Pre-training Tasks for Dense Retrieval](../../wiki/summaries/O%C4%9Fuz%20et%20al.%20-%202021%20-%20Domain-matched%20Pre-training%20Tasks%20for%20Dense%20Retrieval.md)
- 当前角色：RAG 与开放问答相关能力的基础概念

## 相关主张

- Dense Retrieval 用向量空间匹配替代传统稀疏词项匹配。
- 在当前知识库里，它既是 DPR 的上位概念，也与句向量学习路线高度相关。

## 来源支持

- [Oğuz et al. - 2021 - Domain-matched Pre-training Tasks for Dense Retrieval](../../wiki/summaries/O%C4%9Fuz%20et%20al.%20-%202021%20-%20Domain-matched%20Pre-training%20Tasks%20for%20Dense%20Retrieval.md)

## 关联页面

- [DPR](./DPR.md)
- [SimCSE](./SimCSE.md)
- [Sentence-BERT](./Sentence-BERT.md)
- [传统 NLP](../topics/传统%20NLP.md)
