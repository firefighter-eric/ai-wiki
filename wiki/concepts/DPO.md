---
type: concept
---
# DPO

## TL;DR（快速导读）

DPO 利用成对的好坏回答直接训练模型偏好，简化独立奖励模型与在线策略优化的部分流程。

## 简介

DPO 利用成对的好坏回答直接训练模型偏好，简化独立奖励模型与在线策略优化的部分流程。

## 具体怎么理解

同一问题有两个回答，训练推动模型更倾向被选中的回答；标注质量和参考模型仍会影响结果。

## 关键属性

- 类型：偏好优化 / post-training 方法
- 代表来源：
  - [Rafailov et al. - 2023 - Direct Preference Optimization Your Language Model is Secretly a Reward Model](../../wiki/summaries/Rafailov%20et%20al.%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.md)
- 当前角色：RLHF 之后更轻量的对齐方法代表

## 相关主张

- `Rafailov et al. 2023` 认为可把标准 RLHF 问题改写成更直接的优化形式，从而避免复杂的 reward model + RL 管线。
- 在知识库里，DPO 代表“偏好对齐可以更稳定、更简单地实现”的方法论变化。
- 它适合作为对比 InstructGPT / PPO 式 RLHF 的方法概念页。

## 来源支持

- [Rafailov et al. - 2023 - Direct Preference Optimization Your Language Model is Secretly a Reward Model](../../wiki/summaries/Rafailov%20et%20al.%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.md)

## 关联页面

- [InstructGPT](./InstructGPT.md)
- [LLM RL](../topics/LLM%20RL.md)
- [指令对齐与 post-training](../topics/指令对齐与%20post-training.md)

## 这里的术语是什么意思

- **reward model**：奖励模型：根据训练信号给回答或行为打分，分数是目标的近似。
- **RLHF**：人类反馈强化学习：把人类偏好转成奖励，并用它调整模型行为。
- **DPO**：直接偏好优化：用成对偏好数据直接调整模型概率，简化部分奖励训练流程。
