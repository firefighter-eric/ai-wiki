---
type: concept
---
# Prompt Tuning

## TL;DR（快速导读）

提示微调冻结模型主体，只训练连续提示向量，为具体任务提供低参数适配；这些向量不是手写提示词。

## 简介

提示微调冻结模型主体，只训练连续提示向量，为具体任务提供低参数适配；这些向量不是手写提示词。

## 具体怎么理解

为不同分类任务保存各自的小提示参数，同时共享一个底座；任务效果仍受模型与数据限制。

## 关键属性

- 类型：参数高效适配方法
- 代表来源：[Yang et al. - 2022 - Prompt Tuning for Generative Multimodal Pretrained Models](../../wiki/summaries/Yang%20et%20al.%20-%202022%20-%20Prompt%20Tuning%20for%20Generative%20Multimodal%20Pretrained%20Models.md)
- 当前角色：连接 LoRA 与多模态适配方法

## 相关主张

- Prompt Tuning 通过少量可学习提示参数对大模型进行任务适配。
- 在当前知识库里，它补足了 LoRA 之外另一类常见轻量适配方法。

## 来源支持

- [Yang et al. - 2022 - Prompt Tuning for Generative Multimodal Pretrained Models](../../wiki/summaries/Yang%20et%20al.%20-%202022%20-%20Prompt%20Tuning%20for%20Generative%20Multimodal%20Pretrained%20Models.md)

## 关联页面

- [LoRA](./LoRA.md)
- [OFA](./OFA.md)
- [LLM RL](../topics/LLM%20RL.md)
- [指令对齐与 post-training](../topics/指令对齐与%20post-training.md)

## 这里的术语是什么意思

- **LoRA**：低秩适配：冻结底座，只训练较小矩阵表示的权重增量。
