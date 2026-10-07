---
type: concept
---
# HuBERT

## TL;DR（快速导读）

HuBERT 利用语音聚类产生的离散单元做遮挡预测，从未标注音频学习语音表示，之后可适配识别等任务。

## 简介

HuBERT 利用语音聚类产生的离散单元做遮挡预测，从未标注音频学习语音表示，之后可适配识别等任务。

## 具体怎么理解

遮住一段声音，让模型结合前后音频猜测隐藏单元；训练目标并不是直接读取人工转写文字。

## 关键属性

- 类型：语音自监督模型
- 代表来源：[Hsu et al. - 2021 - HuBERT Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](../../wiki/summaries/Hsu%20et%20al.%20-%202021%20-%20HuBERT%20Self-Supervised%20Speech%20Representation%20Learning%20by%20Masked%20Prediction%20of%20Hidden%20Units.md)
- 当前角色：语音基础模型线索的代表页

## 相关主张

- HuBERT 用离散目标与 masked prediction 学习语音表示。
- 在当前知识库里，它与 data2vec 共同体现“自监督基础模型”扩展到语音模态。

## 来源支持

- [Hsu et al. - 2021 - HuBERT Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](../../wiki/summaries/Hsu%20et%20al.%20-%202021%20-%20HuBERT%20Self-Supervised%20Speech%20Representation%20Learning%20by%20Masked%20Prediction%20of%20Hidden%20Units.md)

## 关联页面

- [data2vec](./data2vec.md)
- [Transformer](./Transformer.md)
- [传统 CV](../topics/传统%20CV.md)
