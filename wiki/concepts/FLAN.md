---
type: concept
---
# FLAN

## TL;DR（快速导读）

FLAN 用多任务指令微调改善未见任务的零样本表现，核心是让模型学习如何按照自然语言任务说明工作。

## 简介

FLAN 用多任务指令微调改善未见任务的零样本表现，核心是让模型学习如何按照自然语言任务说明工作。

## 具体怎么理解

训练做过问答与翻译后，再用新任务说明测试；要确认测试任务没有提前混入训练。

## 关键属性

- 类型：指令微调模型
- 代表来源：[Wei et al. - 2021 - Finetuned Language Models Are Zero-Shot Learners](../../wiki/summaries/Wei%20et%20al.%20-%202021%20-%20Finetuned%20Language%20Models%20Are%20Zero-Shot%20Learners.md)
- 当前角色：连接 T5 与 Instruction Tuning 方法层

## 相关主张

- FLAN 说明指令微调能显著改善未见任务的 zero-shot 表现。
- 在当前知识库里，它是 InstructGPT 之前更偏监督式的指令泛化节点。

## 来源支持

- [Wei et al. - 2021 - Finetuned Language Models Are Zero-Shot Learners](../../wiki/summaries/Wei%20et%20al.%20-%202021%20-%20Finetuned%20Language%20Models%20Are%20Zero-Shot%20Learners.md)

## 关联页面

- [T5](./T5.md)
- [Instruction Tuning](./Instruction Tuning.md)
- [InstructGPT](./InstructGPT.md)
- [LLM RL](../topics/LLM%20RL.md)
- [指令对齐与 post-training](../topics/指令对齐与%20post-training.md)
