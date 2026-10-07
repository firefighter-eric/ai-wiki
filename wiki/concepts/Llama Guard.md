---
type: concept
---
# Llama Guard

## TL;DR（快速导读）

Llama Guard 对对话输入或输出进行安全分类，作为系统的一道检查环节；分类器本身也有误判和漏判。

## 简介

Llama Guard 对对话输入或输出进行安全分类，作为系统的一道检查环节；分类器本身也有误判和漏判。

## 具体怎么理解

在生成回答前后检查内容类别，可帮助决定是否继续处理；高分类分数并不保证完整系统不存在风险。

## 关键属性

- 类型：安全模型 / safeguard
- 代表来源：[Inan et al. - 2023 - Llama Guard LLM-based Input-Output Safeguard for Human-AI Conversations](../../wiki/summaries/Inan%20et%20al.%20-%202023%20-%20Llama%20Guard%20LLM-based%20Input-Output%20Safeguard%20for%20Human-AI%20Conversations.md)
- 当前角色：对齐与安全层的重要概念页

## 相关主张

- Llama Guard 把安全审核建模成专门的输入输出分类与判定任务。
- 在当前知识库里，它补足了“模型对齐”之外的“系统防护”层。

## 来源支持

- [Inan et al. - 2023 - Llama Guard LLM-based Input-Output Safeguard for Human-AI Conversations](../../wiki/summaries/Inan%20et%20al.%20-%202023%20-%20Llama%20Guard%20LLM-based%20Input-Output%20Safeguard%20for%20Human-AI%20Conversations.md)

## 关联页面

- [Llama 3](./Llama 3.md)
- [RLHF](./RLHF.md)
- [LLM RL](../topics/LLM%20RL.md)
