---
type: concept
---
# GRPO

## TL;DR（快速导读）

GRPO 对同一问题采样多份回答，利用组内相对奖励调整策略，省去单独价值模型的部分资源成本。

## 简介

GRPO 对同一问题采样多份回答，利用组内相对奖励调整策略，省去单独价值模型的部分资源成本。

## 具体怎么理解

一组回答有好有坏时，奖励差异提供学习信号；如果全都一样，信号与稳定性会受到影响。

## 关键属性

- 类型：在线 RL / policy optimization 方法
- 代表来源：
  - [Shao et al. - 2024 - DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models](../../wiki/summaries/Shao%20et%20al.%20-%202024%20-%20DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models.md)
  - [DeepSeek-R1：奖励驱动推理与多阶段训练（2025）](../../wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.md)
  - [Yu et al. - 2025 - DAPO An Open-Source LLM Reinforcement Learning System at Scale](../../wiki/summaries/Yu%20et%20al.%20-%202025%20-%20DAPO%20An%20Open-Source%20LLM%20Reinforcement%20Learning%20System%20at%20Scale.md)
- 当前角色：连接 `PPO` 式 RLHF 与 reasoning-oriented RL 的关键算法节点

## 相关主张

- `DeepSeekMath` 将 `GRPO` 定义为 `PPO` 的高效变体：不训练 critic，而用同组采样结果的相对分数估计 advantage。
- `DeepSeek-R1` 把 `GRPO` 用作核心 RL 框架，说明它已从数学专门场景进入通用 reasoning model 训练主线。
- `DAPO` 进一步表明，朴素 `GRPO` 在长链路推理 RL 中会暴露 entropy collapse、奖励噪声与稳定性问题，因此 `GRPO` 更像基础骨架，而不是完整工程 recipe。

## 来源支持

- [Shao et al. - 2024 - DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models](../../wiki/summaries/Shao%20et%20al.%20-%202024%20-%20DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models.md)
- [DeepSeek-R1：奖励驱动推理与多阶段训练（2025）](../../wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.md)
- [Yu et al. - 2025 - DAPO An Open-Source LLM Reinforcement Learning System at Scale](../../wiki/summaries/Yu%20et%20al.%20-%202025%20-%20DAPO%20An%20Open-Source%20LLM%20Reinforcement%20Learning%20System%20at%20Scale.md)

## 关联页面

- [DAPO](./DAPO.md)
- [DeepSeek-R1](./DeepSeek-R1.md)
- [RLHF](./RLHF.md)
- [LLM RL](../topics/LLM%20RL.md)
- [Hoppe, Toussaint - 2020 - Qgraph-bounded Q-learning Stabilizing Model-Free Off-Policy Deep Reinforcement Learning](../summaries/Hoppe%2C%20Toussaint%20-%202020%20-%20Qgraph-bounded%20Q-learning%20Stabilizing%20Model-Free%20Off-Policy%20Deep%20Reinforcement%20Learning.md)：邻接 RL：Qgraph 约束 off-policy Q-learning；连续控制结果不作为 LLM 训练证据。

## 这里的术语是什么意思

- **critic**：价值模型：估计状态或行为的预期回报，为策略更新提供参照。
- **GRPO**：组相对策略优化：利用同一问题多份回答的相对奖励进行更新。
