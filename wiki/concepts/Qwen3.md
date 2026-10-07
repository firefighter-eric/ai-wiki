---
type: concept
---
# Qwen3

## TL;DR（快速导读）

Qwen3 引入可切换的思考方式并扩展专家混合与代理能力，关注任务质量与推理预算怎样平衡。

## 简介

Qwen3 引入可切换的思考方式并扩展专家混合与代理能力，关注任务质量与推理预算怎样平衡。

## 具体怎么理解

简单问答未必需要长推理，复杂题目可能需要更多预算；模式切换应结合延迟和正确率评估。

## 关键属性

- 类型：大语言模型家族
- 代表来源：
  - [Qwen Team - 2025 - Qwen3 Think Deeper Act Faster](../../wiki/summaries/Qwen%20Team%20-%202025%20-%20Qwen3%20Think%20Deeper%20Act%20Faster.md)
- 当前角色：Qwen3.5 之前的 reasoning + agent 过渡代

## 相关主张

- Qwen3 的关键设计不是单纯更大，而是把 thinking / non-thinking 双模式做成统一接口。
- 它通过更大预训练规模与 RL/post-training 配方，把 Qwen 从通用问答模型推进到可控的 agent 基座。
- Qwen3 已经在数据构建和能力增强上反向调用 Qwen2.5-VL、Qwen2.5-Math、Qwen2.5-Coder。

## 来源支持

- [Qwen Team - 2025 - Qwen3 Think Deeper Act Faster](../../wiki/summaries/Qwen%20Team%20-%202025%20-%20Qwen3%20Think%20Deeper%20Act%20Faster.md)

## 关联页面

- [Qwen2.5](./Qwen2.5.md)
- [Qwen3.5](./Qwen3.5.md)
- [Qwen](./Qwen.md)
- [Qwen 系列](../topics/Qwen%20系列.md)

## 这里的术语是什么意思

- **agent**：代理：围绕任务读材料、调用工具和连续执行的系统；名称本身不保证自主性或质量。
- **post-training**：后训练：在预训练底座上继续调整指令遵循、偏好或其他行为。
