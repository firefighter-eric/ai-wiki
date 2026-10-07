---
type: concept
---
# RoBERTa

## TL;DR（快速导读）

RoBERTa 系统重做 BERT 的训练配置，说明更多数据与更充分训练可能比新增结构更关键。

## 简介

RoBERTa 系统重做 BERT 的训练配置，说明更多数据与更充分训练可能比新增结构更关键。

## 具体怎么理解

比较新旧方案时，应先确认旧方案是否训练充分，否则无法确定提升来自哪个改动。

## 关键属性

- 类型：预训练语言模型
- 代表来源：[Liu et al. - 2019 - RoBERTa A Robustly Optimized BERT Pretraining Approach](../../wiki/summaries/Liu%20et%20al.%20-%202019%20-%20RoBERTa%20A%20Robustly%20Optimized%20BERT%20Pretraining%20Approach.md)
- 当前角色：BERT 系列的重要训练优化节点

## 相关主张

- RoBERTa 强调训练配方、数据规模与动态 masking 对性能的影响。
- 在当前知识库里，它是 BERT 之后“优化预训练过程”而非重写架构的代表页。

## 来源支持

- [Liu et al. - 2019 - RoBERTa A Robustly Optimized BERT Pretraining Approach](../../wiki/summaries/Liu%20et%20al.%20-%202019%20-%20RoBERTa%20A%20Robustly%20Optimized%20BERT%20Pretraining%20Approach.md)

## 关联页面

- [BERT](./BERT.md)
- [SpanBERT](./SpanBERT.md)
- [BERT类双向Transformer语言模型](../topics/BERT%E7%B1%BB%E5%8F%8C%E5%90%91Transformer%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)
- [传统 NLP](../topics/传统%20NLP.md)
