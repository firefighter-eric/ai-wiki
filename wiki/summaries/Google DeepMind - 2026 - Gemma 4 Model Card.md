# Google DeepMind - 2026 - Gemma 4 Model Card

## 来源信息

- 类型：Google AI for Developers 官方模型卡
- 原始 HTML：[raw/html/Google DeepMind - 2026 - Gemma 4 Model Card.html](../../raw/html/Google%20DeepMind%20-%202026%20-%20Gemma%204%20Model%20Card.html)
- 全文文本：[raw/text/Google DeepMind - 2026 - Gemma 4 Model Card.md](../../raw/text/Google%20DeepMind%20-%202026%20-%20Gemma%204%20Model%20Card.md)
- 来源 URL：https://ai.google.dev/gemma/docs/core/model_card_4
- 作者：Google DeepMind
- 年份：2026
- 状态：已整理

## 摘要

`Gemma 4` 是 Google DeepMind 在 2026 年发布的开放模型家族。模型卡显示，该代 Gemma 不再只是小尺寸开放 LLM，而是覆盖 `E2B / E4B / 12B Unified / 26B A4B MoE / 31B Dense` 的多架构家族，并把长上下文、多模态、thinking mode、function calling、system role 和 agentic coding 能力纳入统一路线。

对 DiffusionGemma 来说，最关键的是 `Gemma 4 26B A4B MoE`：它提供 `25.2B` 总参数、`3.8B` 激活参数、`256K` 上下文和 `128` experts / `8` active experts 的 backbone。DiffusionGemma 不是独立从零训练的文本扩散模型，而是在这个 backbone 上引入 discrete diffusion / denoising generation 路径。

## 关键事实

- 许可：Apache 2.0。
- 家族尺寸：`E2B`、`E4B`、`12B Unified`、`26B A4B`、`31B Dense`。
- 模态：Gemma 4 支持文本和图像；`E2B / E4B / 12B` 还支持音频，`26B A4B` 与 `31B` 不支持音频。
- 上下文：小模型为 `128K`，中等/大模型支持 `256K`。
- 架构：Gemma 4 同时包含 dense 与 `MoE` 变体，并使用 local sliding window attention 与 global attention 交错的混合 attention 设计。
- `26B A4B MoE` 规格：总参数 `25.2B`，激活参数 `3.8B`，`30` 层，sliding window `1024` tokens，上下文 `256K`，词表 `262K`。
- 专家配置：`8 active / 128 total and 1 shared`。
- 支持能力：thinking mode、long context、image understanding、video-as-frames understanding、function calling、coding/reasoning、multilingual。
- 模型卡明确区分了小模型的 `PLE`、12B 的 encoder-free unified 架构、26B A4B 的 MoE 架构。

## 争议与不确定点

- 模型卡主要给出官方 benchmark 与安全评测口径，仍需要第三方复现来判断实际部署质量。
- Gemma 4 的能力是预训练、post-training、工具协议和多模态工程共同作用的结果；不应把所有能力都归因于预训练。
- `26B A4B` 的高效率来自稀疏激活，但部署效果仍取决于 runtime 对 MoE、长上下文和 multimodal preprocessing 的支持。

## 关联页面

- 概念：[Gemma 4](../concepts/Gemma%204.md)
- 概念：[Gemma](../concepts/Gemma.md)
- 概念：[MoE](../concepts/MoE.md)
- 概念：[DiffusionGemma](../concepts/DiffusionGemma.md)
- 主题：[LLM 预训练](../topics/LLM%20预训练.md)

