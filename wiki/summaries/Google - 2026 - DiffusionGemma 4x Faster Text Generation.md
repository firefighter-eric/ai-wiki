# Google - 2026 - DiffusionGemma 4x Faster Text Generation

## 来源信息

- 类型：官方发布博客
- 原始 HTML：[raw/html/Google - 2026 - DiffusionGemma 4x Faster Text Generation.html](../../raw/html/Google%20-%202026%20-%20DiffusionGemma%204x%20Faster%20Text%20Generation.html)
- 全文文本：[raw/text/Google - 2026 - DiffusionGemma 4x Faster Text Generation.md](../../raw/text/Google%20-%202026%20-%20DiffusionGemma%204x%20Faster%20Text%20Generation.md)
- 来源 URL：https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/
- 作者：Brendan O'Donoghue、Sebastian Flennerhag
- 年份：2026
- 状态：已整理

## 摘要

这篇 Google 官方发布博客给出 DiffusionGemma 的产品定位：它是一个实验性开放模型，目标是探索 text diffusion 在本地、低并发、单用户交互场景中的速度优势。博客强调它在专用 GPU 上可达到最高约 `4x` 的文本生成加速，但同时明确说明标准自回归 `Gemma 4` 仍是高质量生产输出的推荐选择。

该来源最重要的价值，是把 DiffusionGemma 从“又一个 Gemma 变体”定位为对自回归解码瓶颈的架构实验：模型通过一次生成一个 `256-token` block，把单用户推理从 memory-bound next-token decoding 推向更能利用 GPU compute 的并行 denoising。

## 关键事实

- 发布时间：2026-06-10。
- 官方称 DiffusionGemma 在专用 GPU 上最高可达 `4x` 更快文本生成。
- 公开性能口径包括：单张 NVIDIA H100 上 `1000+ tokens/sec`，NVIDIA GeForce RTX 5090 上 `700+ tokens/sec`。
- 博客强调速度优势来自把 decode bottleneck 从 memory bandwidth 转向 compute。
- 模型以 `26B` 总参数、`3.8B` 激活参数的 `MoE` 形式运行；量化后可落入高端消费级 GPU 约 `18GB VRAM` 范围。
- 双向注意力和并行 block 生成使它更适合 inline editing、code infilling、amino acid sequences、mathematical graphs 等非线性结构任务。
- 官方明确说 DiffusionGemma 的 overall output quality 低于标准 Gemma 4；若目标是最高质量，仍建议使用标准 Gemma 4。
- Google 将其与 `Gemini Diffusion` 研究相连，并把 DiffusionGemma描述为在 Gemma 4 家族上加入 diffusion head 的开放实验。
- 博客说明 Apple Silicon 这类 unified-memory 架构未必获得同等加速，因为该速度优势依赖高 arithmetic intensity 的专用加速器。

## 争议与不确定点

- `4x` 是低并发、专用 GPU、小 batch 场景下的速度口径，不应理解为任何部署环境都比自回归模型快。
- 博客是发布材料，完整训练细节、模型损失函数和全部复现实验仍需要更技术化来源补充。
- 对高 QPS 云服务来说，自回归模型可通过大 batch 饱和硬件，DiffusionGemma 的并行解码优势会减弱甚至提高服务成本。

## 关联页面

- 概念：[DiffusionGemma](../concepts/DiffusionGemma.md)
- 概念：[Gemma 4](../concepts/Gemma%204.md)
- 主题：[文本扩散语言模型](../topics/%E6%96%87%E6%9C%AC%E6%89%A9%E6%95%A3%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)

