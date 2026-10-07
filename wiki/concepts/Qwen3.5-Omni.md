---
type: concept
---
# Qwen3.5-Omni

## TL;DR（快速导读）

本库的 Qwen3.5-Omni 快照延续端到端音视频交互路线，应按当时资料核对支持模态和具体接口。

## 简介

本库的 Qwen3.5-Omni 快照延续端到端音视频交互路线，应按当时资料核对支持模态和具体接口。

## 具体怎么理解

输入视频并要求语音回应时，需要确认模型版本、是否支持流式以及声音输出的限制。

## 关键属性

- 类型：fully omnimodal LLM
- 代表来源：
  - [Qwen Team - 2026 - Qwen3.5-Omni Scaling Up Toward Native Omni-Modal AGI](../../wiki/summaries/Qwen%20Team%20-%202026%20-%20Qwen3.5-Omni%20Scaling%20Up%20Toward%20Native%20Omni-Modal%20AGI.md)
- 当前角色：本库已收录的 Qwen omni 系列节点

## 相关主张

- Qwen3.5-Omni 已从单一开源示范模型扩展为 `Plus / Flash / Light` 多尺寸系列。
- 相比 Qwen2.5-Omni，它更强调 `Hybrid-Attention MoE`、`256K` 长上下文和超长音频处理。
- 在家族结构上，它说明 omni 分支已从实验节点升级为 Qwen 的正式主线之一。

256K、长音频与视频输入的容量声明要按 Plus/Flash/Light 和采样条件核对。215 项 SOTA 包含 benchmark 与语言子任务，不代表 215 个独立数据集。依据：[官方材料摘要](../summaries/Qwen%20Team%20-%202026%20-%20Qwen3.5-Omni%20Scaling%20Up%20Toward%20Native%20Omni-Modal%20AGI.md)。

## 来源支持

- [Qwen Team - 2026 - Qwen3.5-Omni Scaling Up Toward Native Omni-Modal AGI](../../wiki/summaries/Qwen%20Team%20-%202026%20-%20Qwen3.5-Omni%20Scaling%20Up%20Toward%20Native%20Omni-Modal%20AGI.md)

## 关联页面

- [Qwen3.5](./Qwen3.5.md)
- [Qwen2.5-Omni](./Qwen2.5-Omni.md)
- [Qwen](./Qwen.md)
- [Qwen 系列](../topics/Qwen%20系列.md)

## 这里的术语是什么意思

- **MoE**：专家混合：路由器为不同输入选择部分子网络，总容量与每次实际计算不同。
