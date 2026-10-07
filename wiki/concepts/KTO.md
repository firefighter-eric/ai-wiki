---
type: concept
---
# KTO

## TL;DR（快速导读）

KTO 使用单个回答的好坏反馈进行偏好训练，不强制每条样本都有一对回答，适合研究不同反馈接口。

## 简介

KTO 使用单个回答的好坏反馈进行偏好训练，不强制每条样本都有一对回答，适合研究不同反馈接口。

## 具体怎么理解

只有“这个回答好”或“这个回答不好”的记录时，可提供单项反馈；它与在两个回答中选一个的数据不同。

## 关键属性

- 类型：偏好优化 / human-aware loss
- 代表来源：
  - [Ethayarajh et al. - 2024 - KTO Model Alignment as Prospect Theoretic Optimization](../../wiki/summaries/Ethayarajh%20et%20al.%20-%202024%20-%20KTO%20Model%20Alignment%20as%20Prospect%20Theoretic%20Optimization.md)
- 当前角色：把偏好优化从 pairwise preference 扩展到 unary desirability signal 的代表方法

## 相关主张

- `KTO` 认为高质量对齐未必要求成对偏好比较，只要能判断一个响应是 desirable 还是 undesirable，就能构造有效目标。
- 该方法把 `DPO`、`PPO` 等对齐损失放进 `HALO` 框架，并以 prospect theory 作为设计 `KTO` 的理论动机。
- 在当前知识库里，`KTO` 的重要性在于它改变了后训练的数据接口假设，而不只是对 `DPO` 做小幅公式改写。

## 来源支持

- [Ethayarajh et al. - 2024 - KTO Model Alignment as Prospect Theoretic Optimization](../../wiki/summaries/Ethayarajh%20et%20al.%20-%202024%20-%20KTO%20Model%20Alignment%20as%20Prospect%20Theoretic%20Optimization.md)

## 关联页面

- [DPO](./DPO.md)
- [ORPO](./ORPO.md)
- [RLHF](./RLHF.md)
- [LLM RL](../topics/LLM%20RL.md)

## 这里的术语是什么意思

- **DPO**：直接偏好优化：用成对偏好数据直接调整模型概率，简化部分奖励训练流程。
