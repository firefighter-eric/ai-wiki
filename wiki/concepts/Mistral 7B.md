---
type: concept
---
# Mistral 7B

## TL;DR（快速导读）

Mistral 7B 用较紧凑的密集模型与注意力设计提供开放语言能力，适合研究单位参数效果和部署成本。

## 简介

Mistral 7B 用较紧凑的密集模型与注意力设计提供开放语言能力，适合研究单位参数效果和部署成本。

## 具体怎么理解

比较它与更大模型时，应确认任务、上下文长度和推理配置；参数少不自动代表任何设备上都更快。

## 关键属性

- 类型：开放语言模型
- 开放性：`open-weight`
- 代表来源：[Jiang et al. - 2023 - Mistral 7B](../../wiki/summaries/Jiang%20et%20al.%20-%202023%20-%20Mistral%207B.md)
- 当前角色：Mistral 家族与“小而强” dense 开放模型路线的起点

## 相关主张

- `Mistral 7B` 证明 `7B` 级别模型也可以通过更优架构和训练配方打出强竞争力。
- 它把 `GQA` 与长窗口注意力做成了开放模型工程中的主流组件之一。
- 在当前知识库里，`Mistral 7B` 是从 `LLaMA` 走向 `Mixtral` 之前最重要的效率型开放节点之一。

## 来源支持

- [Jiang et al. - 2023 - Mistral 7B](../../wiki/summaries/Jiang%20et%20al.%20-%202023%20-%20Mistral%207B.md)

## 关联页面

- [Mixtral](./Mixtral.md)
- [Llama 2](./Llama 2.md)
- [Gemma](./Gemma.md)
- [LLM 预训练](../topics/LLM%20预训练.md)
- [开放模型家族与中国重要家族对照](../comparisons/开放模型家族与中国重要家族对照.md)
