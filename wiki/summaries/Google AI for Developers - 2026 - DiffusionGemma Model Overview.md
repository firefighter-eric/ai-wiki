---
type: summary
status: refined
---
# Google AI for Developers - 2026 - DiffusionGemma Model Overview

## 来源信息

- 类型：Google AI for Developers 官方文档
- 原始 HTML：[raw/html/Google AI for Developers - 2026 - DiffusionGemma Model Overview.html](../../raw/html/Google%20AI%20for%20Developers%20-%202026%20-%20DiffusionGemma%20Model%20Overview.html)
- 全文文本：[raw/text/Google AI for Developers - 2026 - DiffusionGemma Model Overview.md](../../raw/text/Google%20AI%20for%20Developers%20-%202026%20-%20DiffusionGemma%20Model%20Overview.md)
- 来源 URL：https://ai.google.dev/gemma/docs/diffusiongemma
- 作者：Google AI for Developers
- 年份：2026
- 状态：已整理

## 摘要

这篇官方文档是 DiffusionGemma 的简明规格和推荐配置入口。它明确把 DiffusionGemma 定义为实验性开放模型：基于 `26B (4B active) MoE Gemma 4`，使用 discrete diffusion 生成文本，支持文本、图像、视频输入并输出文本。

与发布博客相比，该文档更适合支撑三个稳定结论：第一，DiffusionGemma 的目标是本地低并发推理速度，而不是替代标准自回归模型的全部场景；第二，生成过程以 `256-token` canvas 为并行单位；第三，官方推荐的 sampler 配置围绕 entropy bound、temperature schedule 和 adaptive early stopping 展开。

## 关键事实

- 模型定位：experimental open model，探索 text diffusion 生成。
- 基础架构：`26B (4B active) Mixture-of-Experts Gemma 4`。
- 生成机制：block-autoregressive multi-canvas sampling，通过并行 denoising 生成 token block。
- 输入模态：文本、图像、视频；不支持音频输入。
- `MoE` 与量化部署：文档称量化后可适配约 `18GB VRAM` 的消费级 GPU。
- 使用边界：低并发、本地单加速器场景收益最大；高 QPS 云端 batch serving 中收益会减弱。
- 推荐最大 denoising steps：`48`。
- 推荐 temperature schedule：`0.8 -> 0.4` 线性下降。
- 推荐 adaptive early stopping：平均 entropy 低于 `0.005` 且连续两次 denoiser prediction 稳定。
- 推荐 token selection：entropy bound 为 `0.1`，仅接受低熵高置信 token，其余 token 重新加噪。

## 争议与不确定点

- 文档是 overview，不提供完整训练配方与所有评测细节。
- 对“视频输入”的表述需要结合模型卡理解为帧序列处理，不应理解为视频生成。
- `18GB VRAM` 和速度收益依赖量化、硬件与 runtime，不能视为 BF16 checkpoint 的最低需求。

## 关联页面

- 概念：[DiffusionGemma](../concepts/DiffusionGemma.md)
- 主题：[文本扩散语言模型](../topics/%E6%96%87%E6%9C%AC%E6%89%A9%E6%95%A3%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)

