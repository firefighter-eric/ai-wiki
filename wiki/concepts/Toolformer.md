---
type: concept
---
# Toolformer

## TL;DR（快速导读）

Toolformer 研究语言模型如何从训练信号中学会何时调用工具、怎样组织输入并利用结果，是工具使用的早期方法。

## 简介

Toolformer 研究语言模型如何从训练信号中学会何时调用工具、怎样组织输入并利用结果，是工具使用的早期方法。

## 具体怎么理解

遇到算术题时，可先调用计算器，再将返回数值放回文本；选对工具与正确解释结果同样重要。

## 关键属性

- 类型：工具使用方法 / 语言模型增强
- 代表来源：[Schick et al. - 2023 - Toolformer Language Models Can Teach Themselves to Use Tools](../../wiki/summaries/Schick%20et%20al.%20-%202023%20-%20Toolformer%20Language%20Models%20Can%20Teach%20Themselves%20to%20Use%20Tools.md)
- 当前角色：连接基础模型与工具调用能力

## 相关主张

- Toolformer 说明工具使用可通过自监督式数据构造进入模型行为。
- 在当前知识库里，它是理解后续 agent 化趋势的代表概念。

## 来源支持

- [Schick et al. - 2023 - Toolformer Language Models Can Teach Themselves to Use Tools](../../wiki/summaries/Schick%20et%20al.%20-%202023%20-%20Toolformer%20Language%20Models%20Can%20Teach%20Themselves%20to%20Use%20Tools.md)

## 关联页面

- [GPT-3](./GPT-3.md)
- [LLM RL](../topics/LLM%20RL.md)

## 这里的术语是什么意思

- **agent**：代理：围绕任务读材料、调用工具和连续执行的系统；名称本身不保证自主性或质量。
