---
type: concept
---
# Qwen2

## TL;DR（快速导读）

Qwen2 将多语言、长上下文与密集或专家混合结构组织成模型家族，是理解 Qwen 后续演进的一环。

## 简介

Qwen2 将多语言、长上下文与密集或专家混合结构组织成模型家族，是理解 Qwen 后续演进的一环。

## 具体怎么理解

密集模型与专家混合型号的总参数、每次计算和部署方式不同，不能只按名称中的数字比较。

## 关键属性

- 类型：大语言模型家族
- 代表来源：
  - [Qwen Team - 2024 - Hello Qwen2](../../wiki/summaries/Qwen%20Team%20-%202024%20-%20Hello%20Qwen2.md)
- 当前角色：连接 Qwen1.5 与 Qwen2.5/2-VL 的代际骨干

## 相关主张

- Qwen2 以更多语言、更长上下文和更成熟的对齐流程，确立了 Qwen 作为开放模型主线竞争者的地位。
- 57B-A14B 说明 Qwen 在这一代已把 MoE 纳入正式家族，而不只是 dense 扩张。
- Qwen2 同时是 Qwen2-VL 与 Qwen2.5 的直接语言底座。

## 来源支持

- [Qwen Team - 2024 - Hello Qwen2](../../wiki/summaries/Qwen%20Team%20-%202024%20-%20Hello%20Qwen2.md)

## 关联页面

- [Qwen1.5](./Qwen1.5.md)
- [Qwen2-VL](./Qwen2-VL.md)
- [Qwen2.5](./Qwen2.5.md)
- [Qwen 系列](../topics/Qwen%20系列.md)

## 这里的术语是什么意思

- **MoE**：专家混合：路由器为不同输入选择部分子网络，总容量与每次实际计算不同。
