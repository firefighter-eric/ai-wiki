---
type: author
---
# Microsoft Research

## TL;DR（快速导读）

这里连接 Microsoft Research 相关的 LoRA、Kosmos、Florence 和文档理解工作，分别看适配、定位与统一视觉任务。

## 简介

这里连接 Microsoft Research 相关的 LoRA、Kosmos、Florence 和文档理解工作，分别看适配、定位与统一视觉任务。

## 从哪里开始读

- [Hu et al. - 2021 - LoRA Low-Rank Adaptation of Large Language Models](../summaries/Hu%20et%20al.%20-%202021%20-%20LoRA%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models.md)：待精读：LoRA 冻结大模型原有权重，只训练小规模低秩增量，让同一个底座能更便宜地适配不同任务。
- [Peng et al. - 2023 - Kosmos-2 Grounding Multimodal Large Language Models to the World](../summaries/Peng%20et%20al.%20-%202023%20-%20Kosmos-2%20Grounding%20Multimodal%20Large%20Language%20Models%20to%20the%20World.md)：待精读：Kosmos-2 把语言中的对象描述与图像中的位置绑定起来，使模型生成文字时也能指出“说的是哪里”。
- [Xiao et al. - 2023 - Florence-2 Advancing a Unified Representation for a Variety of Vision Tasks](../summaries/Xiao%20et%20al.%20-%202023%20-%20Florence-2%20Advancing%20a%20Unified%20Representation%20for%20a%20Variety%20of%20Vision%20Tasks.md)：待精读：Florence-2 用文字提示指定视觉任务，再生成对应描述或位置等输出，把多种视觉能力放进统一接口。

## 当前覆盖

- 当前已形成多篇 summary 支撑的连续来源链
- 页面性质：机构导航页，不是一级事实来源

## 代表来源

- [LoRA](../summaries/Hu%20et%20al.%20-%202021%20-%20LoRA%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models.md)
- [Kosmos-2](../summaries/Peng%20et%20al.%20-%202023%20-%20Kosmos-2%20Grounding%20Multimodal%20Large%20Language%20Models%20to%20the%20World.md)
- [Florence-2](../summaries/Xiao%20et%20al.%20-%202023%20-%20Florence-2%20Advancing%20a%20Unified%20Representation%20for%20a%20Variety%20of%20Vision%20Tasks.md)

## 关联页面

- [LoRA](../concepts/LoRA.md)
- [Kosmos-2](../concepts/Kosmos-2.md)
- [Florence-2](../concepts/Florence-2.md)

## 这里的术语是什么意思

- **LoRA**：低秩适配：冻结底座，只训练较小矩阵表示的权重增量。
