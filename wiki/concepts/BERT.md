---
type: concept
---
# BERT

## TL;DR（快速导读）

BERT 同时利用左右上下文学习文本表示，适合分类、抽取和问答等理解任务；具体任务通常还需微调。

## 简介

BERT 同时利用左右上下文学习文本表示，适合分类、抽取和问答等理解任务；具体任务通常还需微调。

## 具体怎么理解

“苹果发布手机”和“苹果很甜”中的同一词语，因上下文不同而应有不同表示。

## 关键属性

- 类型：预训练语言模型
- 代表来源：[Devlin et al. - 2019 - BERT Pre-training of deep bidirectional transformers for language understanding](../../wiki/summaries/Devlin%20et%20al.%20-%202019%20-%20BERT%20Pre-training%20of%20deep%20bidirectional%20transformers%20for%20language%20understanding.md)
- 当前角色：连接 Transformer 架构与后续 RoBERTa、SpanBERT 等改进路线

## 相关主张

- BERT 把 masked language modeling 与双向编码器范式推到主流位置。
- 在当前知识库里，BERT 也是理解句向量、检索、抽取与文档理解模型的重要基座。

## 来源支持

- [Devlin et al. - 2019 - BERT Pre-training of deep bidirectional transformers for language understanding](../../wiki/summaries/Devlin%20et%20al.%20-%202019%20-%20BERT%20Pre-training%20of%20deep%20bidirectional%20transformers%20for%20language%20understanding.md)

## 关联页面

- [Transformer](./Transformer.md)
- [RoBERTa](./RoBERTa.md)
- [SpanBERT](./SpanBERT.md)
- [BERT类双向Transformer语言模型](../topics/BERT%E7%B1%BB%E5%8F%8C%E5%90%91Transformer%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)
- [传统 NLP](../topics/传统%20NLP.md)
