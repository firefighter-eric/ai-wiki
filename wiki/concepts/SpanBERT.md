---
type: concept
---
# SpanBERT

## TL;DR（快速导读）

SpanBERT 以连续片段为遮挡单位，用边界表示学习片段内容，适合研究实体、答案跨度与指代表示。

## 简介

SpanBERT 以连续片段为遮挡单位，用边界表示学习片段内容，适合研究实体、答案跨度与指代表示。

## 具体怎么理解

把“北京大学”整体遮住，再从边界恢复它，比随机只遮一个字更接近某些跨度任务。

## 关键属性

- 类型：预训练语言模型
- 代表来源：[Joshi et al. - 2020 - Spanbert Improving pre-training by representing and predicting spans](../../wiki/summaries/Joshi%20et%20al.%20-%202020%20-%20Spanbert%20Improving%20pre-training%20by%20representing%20and%20predicting%20spans.md)
- 当前角色：连接通用预训练与抽取/共指等 span 任务

## 相关主张

- SpanBERT 把 span masking 与边界表示作为核心设计点。
- 在当前知识库里，它说明 BERT 系列可围绕任务结构继续演化，而不只是扩大规模。

## 来源支持

- [Joshi et al. - 2020 - Spanbert Improving pre-training by representing and predicting spans](../../wiki/summaries/Joshi%20et%20al.%20-%202020%20-%20Spanbert%20Improving%20pre-training%20by%20representing%20and%20predicting%20spans.md)

## 关联页面

- [BERT](./BERT.md)
- [RoBERTa](./RoBERTa.md)
- [BERT类双向Transformer语言模型](../topics/BERT%E7%B1%BB%E5%8F%8C%E5%90%91Transformer%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)
- [传统 NLP](../topics/传统%20NLP.md)
