---
type: concept
---
# InstructGPT

## TL;DR（快速导读）

InstructGPT 用示范、偏好和强化学习让语言模型更符合用户意图，说明知识能力与交互行为需要分别训练。

## 简介

InstructGPT 用示范、偏好和强化学习让语言模型更符合用户意图，说明知识能力与交互行为需要分别训练。

## 具体怎么理解

能续写网页的模型不一定能遵循“用三句话回答”；对齐训练针对的是这种行为差异。

## 关键属性

- 类型：指令对齐 / RLHF 模型系列
- 代表来源：
  - [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback](../../wiki/summaries/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md)
- 当前角色：RLHF 主线的早期标志性节点

## 相关主张

- `Ouyang et al. 2022` 用 supervised fine-tuning、偏好排序和 RLHF 把 GPT-3 底座转成更符合用户意图的模型。
- 论文强调，参数更大并不自动等于更 helpful、truthful、harmless。
- 在当前知识库中，InstructGPT 代表“后训练能够显著改变模型交互行为”的关键转折点。

## 来源支持

- [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback](../../wiki/summaries/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md)

## 关联页面

- [GPT-3](./GPT-3.md)
- [DPO](./DPO.md)
- [LLM RL](../topics/LLM%20RL.md)
- [指令对齐与 post-training](../topics/指令对齐与%20post-training.md)

## 这里的术语是什么意思

- **RLHF**：人类反馈强化学习：把人类偏好转成奖励，并用它调整模型行为。
