---
type: concept
---
# Llama 2

## TL;DR（快速导读）

Llama 2 同时提供基础与对话模型，连接预训练语言能力和聊天后训练；两种版本适用方式不同。

## 简介

Llama 2 同时提供基础与对话模型，连接预训练语言能力和聊天后训练；两种版本适用方式不同。

## 具体怎么理解

基础模型主要延续文本，对话模型按交互格式响应；比较时需要确认使用的检查点。

## 关键属性

- 类型：大语言模型家族
- 代表来源：
  - [Touvron et al. - 2023 - Llama 2 Open Foundation and Fine-Tuned Chat Models](../../wiki/summaries/Touvron%20et%20al.%20-%202023%20-%20Llama%202%20Open%20Foundation%20and%20Fine-Tuned%20Chat%20Models.md)
- 当前角色：Llama 家族的“开放 foundation + chat”节点

## 相关主张

- Llama 2 的关键变化是将 chat 对齐模型作为家族正式组成部分公开出来。
- 它把 `SFT + RLHF + safety` 组合流程纳入 Llama 家族主线。
- Code Llama 直接建立在 Llama 2 之上，说明这一代已成为可外溢的通用底座。

## 来源支持

- [Touvron et al. - 2023 - Llama 2 Open Foundation and Fine-Tuned Chat Models](../../wiki/summaries/Touvron%20et%20al.%20-%202023%20-%20Llama%202%20Open%20Foundation%20and%20Fine-Tuned%20Chat%20Models.md)

## 关联页面

- [Llama 家族](./Llama%20家族.md)
- [LLaMA（初代）](./LLaMA%20初代.md)
- [Code Llama](./Code Llama.md)
- [Llama 3](./Llama 3.md)
- [LLM预训练](../topics/LLM%20预训练.md)

## 这里的术语是什么意思

- **SFT**：监督微调：用输入与参考输出继续训练已有模型。
- **RLHF**：人类反馈强化学习：把人类偏好转成奖励，并用它调整模型行为。
