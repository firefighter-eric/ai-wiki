---
type: concept
---
# DeepSeek-R1

## TL;DR（快速导读）

DeepSeek-R1 通过强化学习与多阶段训练增强推理，并提供蒸馏模型；应区分 R1-Zero、R1 和后续版本。

## 简介

DeepSeek-R1 通过强化学习与多阶段训练增强推理，并提供蒸馏模型；应区分 R1-Zero、R1 和后续版本。

## 具体怎么理解

数学题答对与解题过程清楚是不同目标；R1 的训练流程也处理可读性和语言稳定性。

## 关键属性

- 类型：推理强化学习模型
- 代表来源：
  - [DeepSeek-R1：奖励驱动推理与多阶段训练（2025）](../../wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.md)
  - [DeepSeek AI - 2025 - DeepSeek-R1-0528 Release](../../wiki/summaries/DeepSeek%20AI%20-%202025%20-%20DeepSeek-R1-0528%20Release.md)
- 当前角色：LLM RL 主线中从传统 RLHF 走向 reasoning RL 的代表节点

## 相关主张

- `DeepSeek-R1` 把“直接通过 RL 激励推理能力”作为核心主张。
- 相比 InstructGPT 更强调用户意图对齐，DeepSeek-R1 更强调 reasoning 行为本身的形成与蒸馏传播。
- `DeepSeek-R1-0528` 把 R1 主线继续推向更稳定的可用接口，包括 JSON output、function calling 与幻觉降低等更新。
- 在当前知识库里，它是连接 RLHF、DPO 与 reasoning RL 的重要概念节点。

## 来源支持

- [DeepSeek-R1：奖励驱动推理与多阶段训练（2025）](../../wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.md)
- [DeepSeek AI - 2025 - DeepSeek-R1-0528 Release](../../wiki/summaries/DeepSeek%20AI%20-%202025%20-%20DeepSeek-R1-0528%20Release.md)

## 关联页面

- [DeepSeek](./DeepSeek.md)
- [DeepSeek 系列](../topics/DeepSeek%20系列.md)
- [DeepSeek-V3](./DeepSeek-V3.md)
- [InstructGPT](./InstructGPT.md)
- [LLM RL](../topics/LLM%20RL.md)

## 这里的术语是什么意思

- **RLHF**：人类反馈强化学习：把人类偏好转成奖励，并用它调整模型行为。
- **DPO**：直接偏好优化：用成对偏好数据直接调整模型概率，简化部分奖励训练流程。
