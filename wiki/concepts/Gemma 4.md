# Gemma 4

## 简介

`Gemma 4` 是 Google DeepMind 在 2026 年发布的开放模型家族节点。在当前知识库中，它标志着 Gemma 从“小中尺寸开放模型家族”推进到覆盖 dense、MoE、统一多模态和 agentic workflow 的更完整开放家族。

它同时也是 [DiffusionGemma](./DiffusionGemma.md) 的直接 backbone 来源：`DiffusionGemma 26B A4B IT` 并不是脱离 Gemma 家族的独立模型，而是在 `Gemma 4 26B A4B MoE` 的基础上把生成接口改写为 discrete text diffusion。

## 关键属性

- 类型：开放权重多模态模型家族
- 许可：Apache 2.0
- 代表来源：
  - [Google DeepMind - 2026 - Gemma 4 Model Card](../summaries/Google%20DeepMind%20-%202026%20-%20Gemma%204%20Model%20Card.md)
  - [Google - 2026 - Gemma 4 Byte for Byte Most Capable Open Models](../summaries/Google%20-%202026%20-%20Gemma%204%20Byte%20for%20Byte%20Most%20Capable%20Open%20Models.md)
- 家族尺寸：`E2B`、`E4B`、`12B Unified`、`26B A4B MoE`、`31B Dense`
- 当前角色：Gemma 开放家族的多架构、多模态和 agentic workflow 扩展节点

## 相关主张

- `Gemma 4` 说明 Google 的开放模型路线已经不只是 practical small model，而是同时覆盖端侧、小模型、MoE、dense 大模型和统一多模态架构。
- `26B A4B` 是当前知识库中重要的开放 `MoE` 节点：它将总参数和激活参数脱钩，用 `25.2B` 总参数、`3.8B` 激活参数服务推理效率。
- `Gemma 4` 引入 native system prompt、thinking mode、function calling 和长上下文能力，因此它横跨预训练、post-training、工具协议和部署接口，不能只按“预训练模型”理解。
- `DiffusionGemma` 的出现说明 Gemma 家族也开始把开放模型能力迁移到非自回归文本生成接口上。

## 来源支持

- [Google DeepMind - 2026 - Gemma 4 Model Card](../summaries/Google%20DeepMind%20-%202026%20-%20Gemma%204%20Model%20Card.md)
- [Google - 2026 - Gemma 4 Byte for Byte Most Capable Open Models](../summaries/Google%20-%202026%20-%20Gemma%204%20Byte%20for%20Byte%20Most%20Capable%20Open%20Models.md)

## 关联页面

- [Gemma](./Gemma.md)
- [Gemma 3](./Gemma%203.md)
- [DiffusionGemma](./DiffusionGemma.md)
- [MoE](./MoE.md)
- [LLM 预训练](../topics/LLM%20预训练.md)
- [文本扩散语言模型](../topics/%E6%96%87%E6%9C%AC%E6%89%A9%E6%95%A3%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)

