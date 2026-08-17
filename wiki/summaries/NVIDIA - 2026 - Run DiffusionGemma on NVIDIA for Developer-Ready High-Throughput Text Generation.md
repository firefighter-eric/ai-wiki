---
type: summary
status: refined
---
# NVIDIA - 2026 - Run DiffusionGemma on NVIDIA for Developer-Ready High-Throughput Text Generation

## 来源信息

- 类型：NVIDIA 技术博客 / 部署与硬件优化资料
- 原始 HTML：[raw/html/NVIDIA - 2026 - Run DiffusionGemma on NVIDIA for Developer-Ready High-Throughput Text Generation.html](../../raw/html/NVIDIA%20-%202026%20-%20Run%20DiffusionGemma%20on%20NVIDIA%20for%20Developer-Ready%20High-Throughput%20Text%20Generation.html)
- 全文文本：[raw/text/NVIDIA - 2026 - Run DiffusionGemma on NVIDIA for Developer-Ready High-Throughput Text Generation.md](../../raw/text/NVIDIA%20-%202026%20-%20Run%20DiffusionGemma%20on%20NVIDIA%20for%20Developer-Ready%20High-Throughput%20Text%20Generation.md)
- 来源 URL：https://developer.nvidia.com/blog/run-diffusiongemma-on-nvidia-for-developer-ready-high-throughput-text-generation/
- 作者：Anu Srivastava
- 年份：2026
- 状态：已整理

## 摘要

这篇 NVIDIA 技术博客从硬件和部署角度解释 DiffusionGemma。其核心观点是：DiffusionGemma 的 parallel denoising 把生成从传统 LLM 单用户场景中的 memory-bound workload 转成更 compute-bound 的 workload，因此更适合 NVIDIA Tensor Core、RTX、DGX Spark、DGX Station 和 H100/Blackwell 这类硬件。

该来源适合支撑部署事实、NVIDIA 平台支持和 `BF16 / NVFP4` 精度信息，但不应替代 Google 模型卡作为模型结构与能力事实源。

## 关键事实

- NVIDIA 将 DiffusionGemma 描述为 Google DeepMind 开发、NVIDIA 优化的开放模型。
- 该文列出模型概览：Text/Image 输入，`25.2B` 总参数，`3.8B` 激活参数，最高 `256K` tokens 上下文。
- 支持精度：`BF16` 与 `NVFP4`。
- 性能口径：单张 NVIDIA H100 最高 `1000 tokens/sec`，NVIDIA DGX Spark 最高 `150 tokens/sec`，并声称 DGX Station 有最快本地性能。
- 部署入口：Hugging Face Transformers 用于原型验证，vLLM 用于更高吞吐或并发服务。
- NVIDIA 称 DiffusionGemma BF16 checkpoint 已在 Hugging Face 可用，NVFP4 量化 checkpoint 可通过 NVIDIA Model Optimizer 获得。
- 企业部署路径：NVIDIA NIM 提供 OpenAI-compatible API 的容器化推理服务。
- 微调路径：NVIDIA NeMo Framework 可用于特定任务或领域适配。

## 争议与不确定点

- 该来源是 NVIDIA 平台视角，速度与部署建议明显依赖 NVIDIA 硬件和软件栈。
- 文中“higher concurrency / serving costs”一类表述与 Google 官方对高 QPS 云端收益递减的提醒需要合并理解：DiffusionGemma 的核心优势仍在小 batch / 低到中等并发场景。
- NVFP4 相关主张需要结合具体 Blackwell/硬件支持和量化 checkpoint 可用性验证。

## 关联页面

- 概念：[DiffusionGemma](../concepts/DiffusionGemma.md)
- 概念：[Gemma 4](../concepts/Gemma%204.md)
- 主题：[文本扩散语言模型](../topics/%E6%96%87%E6%9C%AC%E6%89%A9%E6%95%A3%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)

