---
type: concept
---
# DiffusionGemma

## 简介

`DiffusionGemma` 是 Google DeepMind 基于 `Gemma 4 26B A4B MoE` 发布的实验性开放权重文本生成模型。它的关键不是“生成图片”，而是把 diffusion 的 iterative denoising 思路迁移到离散文本生成：模型一次处理一个 `256-token canvas`，通过多步 denoising 并行修正整块 token，而不是像传统 LLM 一样逐 token 自回归生成。

在当前知识库中，DiffusionGemma 应被理解为 `Gemma 4` 家族的生成接口实验节点：它牺牲一部分标准 Gemma 4 的输出质量和通用 benchmark 表现，换取低并发、本地单用户交互场景中的高 tokens/sec 和可迭代自修正能力。

## 关键属性

- 类型：开放权重文本扩散语言模型 / 多模态 image-text-to-text 模型
- 许可：Apache 2.0
- 基础模型：[Gemma 4](./Gemma%204.md) `26B A4B MoE`
- 总参数：约 `25.2B`
- 激活参数：约 `3.8B`
- 专家结构：`8 active / 128 total`，另有 `1 shared expert`
- 上下文长度：最高 `256K` tokens
- canvas 长度：`256` tokens
- 主要输入：文本、图像；视频可作为帧序列处理
- 输出：文本
- 代表来源：[Google DeepMind - 2026 - DiffusionGemma 26B A4B IT Model Card](../summaries/Google%20DeepMind%20-%202026%20-%20DiffusionGemma%2026B%20A4B%20IT%20Model%20Card.md)

## 相关主张

- DiffusionGemma 的核心机制是 **block-autoregressive denoising**：一个 canvas 内用双向注意力并行 denoise，多个 canvas 之间再按顺序追加到 `KV cache`。
- 它把单用户本地推理从 memory-bandwidth-bound 的逐 token 生成，部分转向更能利用 GPU tensor cores 的 compute-bound block generation。
- 它最适合速度敏感、低并发、局部交互和结构化填充类任务，而不适合被直接当作最高质量生产输出模型。
- 官方 benchmark 显示 DiffusionGemma 在多数质量指标上低于标准 `Gemma 4 26B A4B`；因此更稳妥的定位是“速度/生成范式实验”，不是“Gemma 4 的全面升级”。
- 它的 sampler 配置成为新可调面：最大 denoising steps、temperature schedule、entropy bound、adaptive stopping 和 re-noising 都比传统 LLM 的常规 decoding 参数更重要。

## 来源支持

- [Google DeepMind - 2026 - DiffusionGemma 26B A4B IT Model Card](../summaries/Google%20DeepMind%20-%202026%20-%20DiffusionGemma%2026B%20A4B%20IT%20Model%20Card.md)
- [Google - 2026 - DiffusionGemma 4x Faster Text Generation](../summaries/Google%20-%202026%20-%20DiffusionGemma%204x%20Faster%20Text%20Generation.md)
- [Google Developers - 2026 - DiffusionGemma The Developer Guide](../summaries/Google%20Developers%20-%202026%20-%20DiffusionGemma%20The%20Developer%20Guide.md)
- [Google AI for Developers - 2026 - DiffusionGemma Model Overview](../summaries/Google%20AI%20for%20Developers%20-%202026%20-%20DiffusionGemma%20Model%20Overview.md)
- [NVIDIA - 2026 - Run DiffusionGemma on NVIDIA for Developer-Ready High-Throughput Text Generation](../summaries/NVIDIA%20-%202026%20-%20Run%20DiffusionGemma%20on%20NVIDIA%20for%20Developer-Ready%20High-Throughput%20Text%20Generation.md)
- [Maarten Grootendorst - 2026 - A Visual Guide to DiffusionGemma](../summaries/Maarten%20Grootendorst%20-%202026%20-%20A%20Visual%20Guide%20to%20DiffusionGemma.md)

## 关联页面

- [Gemma 4](./Gemma%204.md)
- [Gemma](./Gemma.md)
- [MoE](./MoE.md)
- [文本扩散语言模型](../topics/%E6%96%87%E6%9C%AC%E6%89%A9%E6%95%A3%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)
- [LLM 预训练](../topics/LLM%20预训练.md)

