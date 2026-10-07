---
type: author
---
# Google Research

## TL;DR（快速导读）

这里连接 Google Research 相关的 Transformer、BERT、T5 与 PaLM 等来源，可沿架构、理解和生成路线阅读。

## 简介

这里连接 Google Research 相关的 Transformer、BERT、T5 与 PaLM 等来源，可沿架构、理解和生成路线阅读。

## 从哪里开始读

- [Sutskever, Vinyals, Le - 2014 - Sequence to Sequence Learning with Neural Networks](../summaries/Sutskever,%20Vinyals,%20Le%20-%202014%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.md)：早期 Seq2Seq 用编码器读取输入序列，再用解码器逐步生成输出，为翻译等任务建立统一接口。
- [Vaswani et al. - 2017 - Attention is all you need](../summaries/Vaswani%20et%20al.%20-%202017%20-%20Attention%20is%20all%20you%20need.md)：Transformer 用注意力与位置编码处理序列，让各位置直接获取其他位置的信息，取代循环计算作为主要结构。
- [Devlin et al. - 2019 - BERT Pre-training of deep bidirectional transformers for language understanding](../summaries/Devlin%20et%20al.%20-%202019%20-%20BERT%20Pre-training%20of%20deep%20bidirectional%20transformers%20for%20language%20understanding.md)：待精读：BERT 通过同时利用词语左右两边的上下文进行预训练，为分类、问答和抽取提供可微调的语言表示。

## 当前覆盖

- 当前已形成多篇 summary 支撑的连续来源链
- 现覆盖从早期 `Seq2Seq` 神经机器翻译、Transformer、BERT 到 T5 的 NLP 架构与任务接口演进
- 页面性质：机构导航页，不是一级事实来源

## 代表来源

- [Sequence to Sequence Learning with Neural Networks](../summaries/Sutskever,%20Vinyals,%20Le%20-%202014%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.md)
- [Attention is all you need](../summaries/Vaswani%20et%20al.%20-%202017%20-%20Attention%20is%20all%20you%20need.md)
- [BERT](../summaries/Devlin%20et%20al.%20-%202019%20-%20BERT%20Pre-training%20of%20deep%20bidirectional%20transformers%20for%20language%20understanding.md)
- [T5](../summaries/Raffel%20et%20al.%20-%202020%20-%20Exploring%20the%20limits%20of%20transfer%20learning%20with%20a%20unified%20text-to-text%20transformer.md)

## 关联页面

- [Transformer](../concepts/Transformer.md)
- [Seq2Seq](../concepts/Seq2Seq.md)
- [BERT](../concepts/BERT.md)
- [T5](../concepts/T5.md)
- [PaLM](../concepts/PaLM.md)
