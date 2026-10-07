---
type: concept
---
# DeepSeek-V3

## TL;DR（快速导读）

DeepSeek-V3 是高效专家混合语言模型，重点技术包括稀疏计算与训练安排；它与 R1 的推理后训练应分开阅读。

## 简介

DeepSeek-V3 是高效专家混合语言模型，重点技术包括稀疏计算与训练安排；它与 R1 的推理后训练应分开阅读。

## 具体怎么理解

先训练通用语言底座，再做推理或指令适配，是不同阶段；模型名称不能代替训练阶段的说明。

## 关键属性

- 类型：MoE 语言模型
- 代表来源：
  - [Unknown - 2024 - DeepSeek-V3 Technical Report](../../wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-V3%20Technical%20Report.md)
- 当前角色：DeepSeek 家族中的预训练与工程能力基座

## 相关主张

- `DeepSeek-V3 Technical Report` 强调其采用 MoE、MLA 和多 token 预测目标来兼顾效率与性能。
- 当前知识库把它视为“高效预训练工程与开放模型能力并进”的代表。
- 它也是理解 DeepSeek-R1 为何能继续沿同一家族推进 reasoning RL 的前置概念。

## 来源支持

- [Unknown - 2024 - DeepSeek-V3 Technical Report](../../wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-V3%20Technical%20Report.md)

## 关联页面

- [DeepSeek](./DeepSeek.md)
- [DeepSeek 系列](../topics/DeepSeek%20系列.md)
- [DeepSeek-R1](./DeepSeek-R1.md)
- [LLM 预训练](../topics/LLM%20预训练.md)

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **MoE**：专家混合：路由器为不同输入选择部分子网络，总容量与每次实际计算不同。
