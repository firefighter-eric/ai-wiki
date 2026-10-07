---
type: concept
---
# OFA

## TL;DR（快速导读）

OFA 用统一序列到序列框架表达多种视觉与语言任务，让任务指令决定输入如何转成输出。

## 简介

OFA 用统一序列到序列框架表达多种视觉与语言任务，让任务指令决定输入如何转成输出。

## 具体怎么理解

同一图像可被要求生成描述或定位对象；统一接口便于复用，评价标准仍要随任务改变。

## 关键属性

- 类型：多模态统一模型
- 代表来源：[Wang et al. - 2022 - OFA Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework](../../wiki/summaries/Wang%20et%20al.%20-%202022%20-%20OFA%20Unifying%20Architectures,%20Tasks,%20and%20Modalities%20Through%20a%20Simple%20Sequence-to-Sequence%20Learning%20Framework.md)
- 当前角色：统一多模态接口的代表页

## 相关主张

- OFA 将图像、文本与多种任务收敛为单一序列到序列问题。
- 在当前知识库里，它与 T5 共享“统一接口”的方法论精神，但跨越了模态边界。
- 从方法谱系看，OFA 是 `Seq2Seq` 从文本条件生成扩展到跨模态任务统一的代表节点。

## 来源支持

- [Sutskever, Vinyals, Le - 2014 - Sequence to Sequence Learning with Neural Networks](../../wiki/summaries/Sutskever,%20Vinyals,%20Le%20-%202014%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.md)
- [Wang et al. - 2022 - OFA Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework](../../wiki/summaries/Wang%20et%20al.%20-%202022%20-%20OFA%20Unifying%20Architectures,%20Tasks,%20and%20Modalities%20Through%20a%20Simple%20Sequence-to-Sequence%20Learning%20Framework.md)

## 关联页面

- [Seq2Seq](./Seq2Seq.md)
- [T5](./T5.md)
- [Florence-2](./Florence-2.md)
- [传统 CV](../topics/传统%20CV.md)
