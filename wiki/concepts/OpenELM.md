---
type: concept
---
# OpenELM

## TL;DR（快速导读）

OpenELM 是面向较小模型效率的开放模型路线，关注层间参数分配、训练与部署，而不只追求总规模。

## 简介

OpenELM 是面向较小模型效率的开放模型路线，关注层间参数分配、训练与部署，而不只追求总规模。

## 具体怎么理解

不同层可以承担不同容量，整体参数相近的模型也可能有不同结构和设备效率。

## 关键属性

- 类型：开放语言模型家族节点
- 开放性：`open-weight`
- 代表来源：[Mehta et al. - 2024 - OpenELM An Efficient Language Model Family with Open Training and Inference Framework](../../wiki/summaries/Mehta%20et%20al.%20-%202024%20-%20OpenELM%20An%20Efficient%20Language%20Model%20Family%20with%20Open%20Training%20and%20Inference%20Framework.md)
- 当前角色：端侧效率型开放模型代表

## 相关主张

- `OpenELM` 强调的不只是模型开放，还包括训练与推理框架开放。
- 它说明端侧效率和开放模型可以被同时当作一等目标。
- 在当前知识库里，`OpenELM` 与 `Phi-3`、`Gemma` 一起构成轻量开放模型的重要对照。

## 来源支持

- [Mehta et al. - 2024 - OpenELM An Efficient Language Model Family with Open Training and Inference Framework](../../wiki/summaries/Mehta%20et%20al.%20-%202024%20-%20OpenELM%20An%20Efficient%20Language%20Model%20Family%20with%20Open%20Training%20and%20Inference%20Framework.md)

## 关联页面

- [Phi-3](./Phi-3.md)
- [Gemma](./Gemma.md)
- [MiniCPM](./MiniCPM.md)
- [LLM 预训练](../topics/LLM%20预训练.md)
- [开放模型家族与中国重要家族对照](../comparisons/开放模型家族与中国重要家族对照.md)
