---
type: concept
---
# LLaMA（初代）

## TL;DR（快速导读）

初代 LLaMA 是 2023 年的开放权重语言模型，研究重点是数据与训练配置如何支持不同尺寸的基础模型。

## 简介

`LLaMA` 是 Meta 在开放权重大模型主线上的第一代关键节点。与 [Llama 家族](./Llama%20家族.md) 总入口不同，本页只对应 2023 年初代 `LLaMA` 模型本身，关注其“高质量开放基础模型”这一历史角色。

## 具体怎么理解

阅读早期 LLaMA 时，不应把 Llama 2 的对话训练或 Llama 3 的能力直接套进去；代际证据应各自对应。

## 关键属性

- 类型：基础语言模型
- 家族位置：Llama 家族初代节点
- 代表来源：[Touvron et al. - 2023 - LLaMA Open and Efficient Foundation Language Models](../summaries/Touvron%20et%20al.%20-%202023%20-%20LLaMA%20Open%20and%20Efficient%20Foundation%20Language%20Models.md)

## 相关主张

- `LLaMA` 证明开放权重模型可以在相对克制的训练预算下逼近当时的闭源前沿能力。
- 它的重要性不只在 benchmark，也在于重新打开了“开放基础模型家族可以持续演进”的路线。
- `Llama 2`、`Code Llama`、`Llama 3` 在这一初代节点之后继续扩展开放模型家族。

## 来源支持

- [Touvron et al. - 2023 - LLaMA Open and Efficient Foundation Language Models](../summaries/Touvron%20et%20al.%20-%202023%20-%20LLaMA%20Open%20and%20Efficient%20Foundation%20Language%20Models.md)

## 关联页面

- [Llama 家族](./Llama%20家族.md)
- [Llama 2](./Llama%202.md)
- [Code Llama](./Code%20Llama.md)
- [Llama 3](./Llama%203.md)
- [LLM 预训练](../topics/LLM%20预训练.md)

## 这里的术语是什么意思

- **benchmark**：评测基准：特定数据、任务与规则的组合，分数只在这些条件下成立。
