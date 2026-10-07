---
type: concept
---
# DAPO

## TL;DR（快速导读）

DAPO 将推理强化学习中的优化与工程技巧组织成训练方案，关注长回答、样本筛选和训练稳定性。

## 简介

DAPO 将推理强化学习中的优化与工程技巧组织成训练方案，关注长回答、样本筛选和训练稳定性。

## 具体怎么理解

同一批题目若全答对或全答错，组内奖励可能缺少区分；有效采样与算法目标需要配合。

## 关键属性

- 类型：reasoning-oriented RL / large-scale training system
- 代表来源：
  - [Yu et al. - 2025 - DAPO An Open-Source LLM Reinforcement Learning System at Scale](../../wiki/summaries/Yu%20et%20al.%20-%202025%20-%20DAPO%20An%20Open-Source%20LLM%20Reinforcement%20Learning%20System%20at%20Scale.md)
- 当前角色：把 `GRPO`-style reasoning RL 推进到更可复现、系统化工程实现的代表节点

## 相关主张

- `DAPO` 从 naive `GRPO` 的失败案例出发，认为长 CoT RL 的主要难点在于熵塌缩、奖励噪声、长度偏置和梯度退化。
- 它通过 `Clip-Higher`、`Dynamic Sampling`、`Token-Level Policy Gradient Loss`、`Overlong Reward Shaping` 等机制修正基础 `GRPO`。
- 在当前知识库里，`DAPO` 的意义不只是“又一个 RL 算法”，而是公开了大规模 reasoning RL 的工程 recipe。

## 来源支持

- [Yu et al. - 2025 - DAPO An Open-Source LLM Reinforcement Learning System at Scale](../../wiki/summaries/Yu%20et%20al.%20-%202025%20-%20DAPO%20An%20Open-Source%20LLM%20Reinforcement%20Learning%20System%20at%20Scale.md)

## 关联页面

- [GRPO](./GRPO.md)
- [DeepSeek-R1](./DeepSeek-R1.md)
- [RLHF](./RLHF.md)
- [LLM RL](../topics/LLM%20RL.md)

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **GRPO**：组相对策略优化：利用同一问题多份回答的相对奖励进行更新。
- **CoT**：思维链：模型写出的中间推理文本，不能自动视为忠实的内部思考。
