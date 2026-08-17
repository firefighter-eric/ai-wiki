---
type: summary
status: refined
---
# Google DeepMind - 2026 - DiffusionGemma 26B A4B IT Model Card

## 来源信息

- 类型：Hugging Face 模型卡 / 官方模型资料
- 原始 HTML：[raw/html/Google DeepMind - 2026 - DiffusionGemma 26B A4B IT Model Card.html](../../raw/html/Google%20DeepMind%20-%202026%20-%20DiffusionGemma%2026B%20A4B%20IT%20Model%20Card.html)
- 全文文本：[raw/text/Google DeepMind - 2026 - DiffusionGemma 26B A4B IT Model Card.md](../../raw/text/Google%20DeepMind%20-%202026%20-%20DiffusionGemma%2026B%20A4B%20IT%20Model%20Card.md)
- 来源 URL：https://huggingface.co/google/diffusiongemma-26B-A4B-it
- 作者：Google DeepMind
- 年份：2026
- 状态：已整理

## 摘要

`DiffusionGemma 26B A4B IT` 是 Google DeepMind 发布在 Hugging Face 上的实验性开放权重模型。它基于 `Gemma 4 26B A4B` 的 `MoE` 架构，但生成方式从传统自回归 next-token decoding 改为离散文本扩散：模型先生成一个含随机 token 的 `canvas`，再通过多步 denoising 并行修正整块文本。

该模型的核心意义不是成为更高质量的 Gemma 4 替代品，而是探索一种更适合本地、低并发、低延迟生成场景的文本生成范式。官方模型卡明确给出：标准 Gemma 4 在多数质量 benchmark 上仍更强，而 DiffusionGemma 的优势主要来自小 batch 条件下的 parallel denoising、adaptive inference time computation 与高单用户 tokens/sec。

## 关键事实

- 许可与开放性：Apache 2.0，开放权重。
- 架构来源：基于 `Gemma 4 26B A4B` 的 `Mixture-of-Experts` 架构。
- 参数规模：总参数约 `25.2B`，激活参数约 `3.8B`。
- 专家结构：`128` 个专家中激活 `8` 个，另有 `1` 个 shared expert。
- 上下文：最高 `256K` tokens。
- `canvas` 长度：`256` tokens。
- 模态：支持文本、图像输入；模型卡正文还说明可把视频按帧处理为输入并生成文本输出，但规格表中主要列为 `Text, Image`，因此视频更应理解为帧序列处理能力，而不是独立视频生成能力。
- 机制：encoder 负责 prompt prefill 与 `KV cache`，denoiser 对 generation canvas 使用双向注意力并行修正 token。
- 采样建议：默认使用 `Entropy-Bounded Denoising` 与 adaptive stopping；最大 denoising steps 为 `48`，温度从 `0.8` 线性降到 `0.4`，entropy bound 为 `0.1`，停止条件包括平均 entropy 低于 `0.005` 且连续两步预测稳定。
- 使用接口：Transformers 中使用 `DiffusionGemmaForBlockDiffusion` 与 `AutoProcessor`。
- 官方 benchmark 显示，DiffusionGemma 在多数通用、视觉和长上下文指标上低于 `Gemma 4 26B A4B`；因此它更像速度/交互实验模型，而不是质量上限模型。

## 争议与不确定点

- 官方明确称其为 experimental model，不能直接按生产级文本质量替代标准 `Gemma 4`。
- 模型卡中同时出现“text, image, video inputs”和规格表“Text, Image”的表述；当前更稳妥的解释是它支持把视频作为帧序列输入，但不是原生视频生成模型。
- 速度口径与硬件、精度、batch size、sampler 配置强相关，不能脱离具体部署环境泛化。

## 关联页面

- 概念：[DiffusionGemma](../concepts/DiffusionGemma.md)
- 概念：[Gemma 4](../concepts/Gemma%204.md)
- 概念：[MoE](../concepts/MoE.md)
- 主题：[文本扩散语言模型](../topics/%E6%96%87%E6%9C%AC%E6%89%A9%E6%95%A3%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)

