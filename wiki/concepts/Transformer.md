# Transformer

## 简介

Transformer 是现代大模型与视觉/语音基础模型的核心架构。在当前知识库中，它是多条技术脉络共享的底层概念。

## 关键属性

- 类型：模型架构
- 代表来源：[Vaswani et al. - 2017 - Attention is all you need](../../wiki/summaries/Vaswani%20et%20al.%20-%202017%20-%20Attention%20is%20all%20you%20need.md)
- 当前角色：连接 NLP、CV、语音和多模态路线的共同基底

## 相关主张

- Transformer 用自注意力替代传统序列建模结构，成为后续大模型共同骨架。
- Transformer 最初仍保留了 `Seq2Seq` 的 encoder-decoder 条件生成接口，只是用 self-attention 与 encoder-decoder attention 替换 RNN 循环。
- 在当前知识库里，许多 concept 页都可回溯到这一架构起点。
- 若要区分标准 attention、线性 attention、稀疏 attention、`MQA / GQA / MLA` 与 `FlashAttention` 等分支，应进入专门的 topic 页，而不是把这些差异折叠进一个概念定义。

## 来源支持

- [Vaswani et al. - 2017 - Attention is all you need](../../wiki/summaries/Vaswani%20et%20al.%20-%202017%20-%20Attention%20is%20all%20you%20need.md)
- [Sutskever, Vinyals, Le - 2014 - Sequence to Sequence Learning with Neural Networks](../../wiki/summaries/Sutskever,%20Vinyals,%20Le%20-%202014%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.md)

## 关联页面

- [Seq2Seq](./Seq2Seq.md)
- [BERT](./BERT.md)
- [GPT-3](./GPT-3.md)
- [ViT](./ViT.md)
- [注意力机制 Attention](../topics/注意力机制%20Attention.md)
- [传统 NLP](../topics/传统%20NLP.md)
- [传统 CV](../topics/传统%20CV.md)
