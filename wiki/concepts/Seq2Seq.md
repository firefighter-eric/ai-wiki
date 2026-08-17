---
type: concept
---
# Seq2Seq

## 简介

`Seq2Seq`（sequence-to-sequence learning）是一种把可变长度输入序列映射到可变长度输出序列的建模范式。它最初在神经机器翻译中成为关键接口，后来扩展到摘要、问答、语音、OCR、多模态统一建模与 text-to-text 预训练。在当前知识库中，`Seq2Seq` 不是某个具体模型名，而是连接 RNN encoder-decoder、Transformer encoder-decoder、T5 和 OFA 的任务接口概念。

## 关键属性

- 类型：序列建模范式 / 条件生成接口
- 代表来源：[Sutskever, Vinyals, Le - 2014 - Sequence to Sequence Learning with Neural Networks](../../wiki/summaries/Sutskever,%20Vinyals,%20Le%20-%202014%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.md)
- 当前角色：解释机器翻译、摘要、text-to-text 预训练与多模态统一生成任务之间的共同结构

## 相关主张

- `Seq2Seq` 的核心是把目标序列写成以源序列为条件的自回归生成：编码源序列，再逐步生成目标 token。
- 2014 年的 RNN / LSTM 版本依赖固定维度向量承载源句信息，这使结构简洁，也形成了后来 attention 与 cross-attention 要解决的信息瓶颈。
- Transformer 并没有取消 seq2seq 接口，而是把 RNN 循环换成 self-attention 与 encoder-decoder attention，让并行训练和长距离依赖建模更有效。
- T5 将 seq2seq 从机器翻译接口提升为通用 text-to-text 任务格式；OFA 则把这种接口进一步扩展到跨模态任务统一。
- `Seq2Seq` 与 decoder-only language model 有历史和结构差异：前者通常显式区分 source encoding 与 target decoding，后者则把条件和输出拼入同一个自回归上下文。

## 来源支持

- [Sutskever, Vinyals, Le - 2014 - Sequence to Sequence Learning with Neural Networks](../../wiki/summaries/Sutskever,%20Vinyals,%20Le%20-%202014%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.md)
- [Vaswani et al. - 2017 - Attention is all you need](../../wiki/summaries/Vaswani%20et%20al.%20-%202017%20-%20Attention%20is%20all%20you%20need.md)
- [Raffel et al. - 2020 - Exploring the limits of transfer learning with a unified text-to-text transformer](../../wiki/summaries/Raffel%20et%20al.%20-%202020%20-%20Exploring%20the%20limits%20of%20transfer%20learning%20with%20a%20unified%20text-to-text%20transformer.md)
- [Wang et al. - 2022 - OFA Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework](../../wiki/summaries/Wang%20et%20al.%20-%202022%20-%20OFA%20Unifying%20Architectures,%20Tasks,%20and%20Modalities%20Through%20a%20Simple%20Sequence-to-Sequence%20Learning%20Framework.md)

## 关联页面

- [Transformer](./Transformer.md)
- [T5](./T5.md)
- [mT5](./mT5.md)
- [OFA](./OFA.md)
- [传统 NLP](../topics/传统%20NLP.md)
- [LLM 预训练](../topics/LLM%20预训练.md)
