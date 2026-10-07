---
type: concept
---
# ColBERT

## TL;DR（快速导读）

ColBERT 提前编码文档的多个词元，在查询时进行较细粒度的相似度交互，折中搜索效果与计算成本。

## 简介

`ColBERT` 是搜索排序中的 late interaction 代表模型。它试图在 cross-encoder 的高效果与 bi-encoder 的高效率之间找到中间解。

## 具体怎么理解

查询中不同词可以分别与文档中最匹配的词比较，再汇总得分；它没有把整份文档压成一个向量就结束。

## 关键属性

- 类型：排序 / 检索模型
- 代表来源：[Khattab, Zaharia - 2020 - ColBERT Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](../../wiki/summaries/Khattab,%20Zaharia%20-%202020%20-%20ColBERT%20Efficient%20and%20Effective%20Passage%20Search%20via%20Contextualized%20Late%20Interaction%20over%20BERT.md)
- 当前角色：连接 `reranking` 与 `dense retrieval` 的关键中间路线

## 相关主张

- `ColBERT` 通过 query/document 分别编码加 token 级 late interaction，保留细粒度匹配能力。
- 它不是传统 cross-encoder，也不是普通双塔，而是以 `MaxSim` 为核心的中间形态。
- 在当前知识库里，`ColBERT` 说明搜索排序的设计空间不只分成“候选重排”与“向量召回”两端。

## 来源支持

- [Khattab, Zaharia - 2020 - ColBERT Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](../../wiki/summaries/Khattab,%20Zaharia%20-%202020%20-%20ColBERT%20Efficient%20and%20Effective%20Passage%20Search%20via%20Contextualized%20Late%20Interaction%20over%20BERT.md)
- [Pretrained Transformers for Text Ranking: BERT and Beyond（Lin、Nogueira、Yates）](../../wiki/summaries/Nogueira%20et%20al.%20-%202020%20-%20Pretrained%20Transformers%20for%20Text%20Ranking%20BERT%20and%20Beyond.md)

## 关联页面

- [搜索排序](../topics/搜索排序.md)
- [Dense Retrieval](./Dense%20Retrieval.md)
- [DPR](./DPR.md)
- [BERT类双向Transformer语言模型](../topics/BERT类双向Transformer语言模型.md)

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **cross-encoder**：交叉编码器：把查询和候选一起输入模型，进行较细的交互比较。
- **bi-encoder**：双塔编码器：分别编码查询和候选，便于提前保存候选向量。
- **late interaction**：后期交互：先独立编码，再在查询时比较多个局部表示。
- **encoder**：编码器：把输入转成模型内部表示。
