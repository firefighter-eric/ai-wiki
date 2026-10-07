---
type: concept
---
# Qwen3.5

## TL;DR（快速导读）

本库的 Qwen3.5 资料将多模态与代理执行放到主干模型中，阅读重点是训练机制和实际任务接口。

## 简介

本库的 Qwen3.5 资料将多模态与代理执行放到主干模型中，阅读重点是训练机制和实际任务接口。

## 具体怎么理解

能分析图片与能连续调用工具完成任务是两类能力；应分别查看原文与应用测试。

## 关键属性

- 类型：原生多模态 / agent 导向模型家族
- 代表来源：
  - [Qwen Team - 2026 - Qwen3.5 Towards Native Multimodal Agents](../../wiki/summaries/Qwen%20Team%20-%202026%20-%20Qwen3.5%20Towards%20Native%20Multimodal%20Agents.md)
- 当前角色：Qwen 家族中从 LLM/VL 分支汇合为 native multimodal agent 的节点

## 相关主张

- 官方已不把 Qwen3.5 描述为单纯文本 LLM，而是直接称为 `native vision-language model`。
- 公开权重的首个代表模型为 `Qwen3.5-397B-A17B`，说明该代已把大规模 MoE 与多模态 agent 合并。
- 与 Qwen3 相比，Qwen3.5 的主线重心明显更偏向 multimodal understanding 与 agent capabilities 的统一。

公开发布说明区分开放权重 397B-A17B 和 Plus API；1M 上下文与官方工具是 Plus API 的声明，不能继承给全部权重型号。依据：[官方发布摘要](../summaries/Qwen%20Team%20-%202026%20-%20Qwen3.5%20Towards%20Native%20Multimodal%20Agents.md)。

## 来源支持

- [Qwen Team - 2026 - Qwen3.5 Towards Native Multimodal Agents](../../wiki/summaries/Qwen%20Team%20-%202026%20-%20Qwen3.5%20Towards%20Native%20Multimodal%20Agents.md)

## 关联页面

- [Qwen3](./Qwen3.md)
- [Qwen3.5-Omni](./Qwen3.5-Omni.md)
- [Qwen](./Qwen.md)
- [Qwen 系列](../topics/Qwen%20系列.md)

## 这里的术语是什么意思

- **MoE**：专家混合：路由器为不同输入选择部分子网络，总容量与每次实际计算不同。
- **agent**：代理：围绕任务读材料、调用工具和连续执行的系统；名称本身不保证自主性或质量。
