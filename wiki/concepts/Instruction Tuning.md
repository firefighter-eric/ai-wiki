---
type: concept
---
# Instruction Tuning

## TL;DR（快速导读）

指令微调把任务说明与参考回答作为训练样本，让预训练模型学会按要求完成任务，是常见的后训练阶段。

## 简介

指令微调把任务说明与参考回答作为训练样本，让预训练模型学会按要求完成任务，是常见的后训练阶段。

## 具体怎么理解

样本可以是“总结这段文章”及其摘要；需要同时检查指令覆盖、答案质量和新任务泛化。

## 关键属性

- 类型：post-training / 指令微调方法
- 代表来源：
  - [Wei et al. - 2021 - Finetuned Language Models Are Zero-Shot Learners](../../wiki/summaries/Wei%20et%20al.%20-%202021%20-%20Finetuned%20Language%20Models%20Are%20Zero-Shot%20Learners.md)
  - [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback](../../wiki/summaries/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md)
- 当前角色：把预训练能力转成可用交互能力的早期通用方法节点

## 相关主张

- `Wei et al. 2021` 表明，在多任务自然语言指令数据上微调可以显著改善模型对未见任务的 zero-shot 泛化。
- 在当前知识库中，Instruction Tuning 不是 RLHF 的替代，而是其前置层或相邻层，用于先把模型行为拉向“遵循指令”。
- `Ouyang et al. 2022` 里的 supervised fine-tuning 阶段也可被视为更完整对齐管线中的 instruction tuning 组成部分。

## 来源支持

- [Wei et al. - 2021 - Finetuned Language Models Are Zero-Shot Learners](../../wiki/summaries/Wei%20et%20al.%20-%202021%20-%20Finetuned%20Language%20Models%20Are%20Zero-Shot%20Learners.md)
- [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback](../../wiki/summaries/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md)

## 关联页面

- [InstructGPT](./InstructGPT.md)
- [RLHF](./RLHF.md)
- [LLM RL](../topics/LLM%20RL.md)
- [LLM 预训练](../topics/LLM%20预训练.md)
- [指令对齐与 post-training](../topics/指令对齐与%20post-training.md)

## 这里的术语是什么意思

- **RLHF**：人类反馈强化学习：把人类偏好转成奖励，并用它调整模型行为。
