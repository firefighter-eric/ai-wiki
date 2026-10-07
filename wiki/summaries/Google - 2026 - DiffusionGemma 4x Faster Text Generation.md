---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Google - 2026 - DiffusionGemma 4x Faster Text Generation

## TL;DR（快速导读）

DiffusionGemma 发布博客介绍文本扩散的速度实验，主要针对本地、低并发的交互场景。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

可以把生成过程理解为先形成一段待定文本，再多轮完善；这是生成机制示意，不能直接推断任何任务都更快。

## 来源信息

- 类型：官方发布博客
- 原始 HTML：[raw/html/Google - 2026 - DiffusionGemma 4x Faster Text Generation.html](../../raw/html/Google%20-%202026%20-%20DiffusionGemma%204x%20Faster%20Text%20Generation.html)
- 全文文本：[raw/text/Google - 2026 - DiffusionGemma 4x Faster Text Generation.md](../../raw/text/Google%20-%202026%20-%20DiffusionGemma%204x%20Faster%20Text%20Generation.md)
- 来源 URL：https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/
- 作者：Brendan O'Donoghue、Sebastian Flennerhag
- 年份：2026
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

模型按块并行修正文本，而非逐个位置顺序生成。官方将它定位为实验性开放模型，并同时讨论质量限制；博客中的速度结果依赖硬件与请求条件，不能外推到所有服务负载。

## 关键事实

- **C1**：DiffusionGemma 是基于 Gemma4 的实验开放 MoE，加入 diffusion head 并以 block 并行生成。
- **C2**：官方报告 H100 超过 1000 tokens/s、RTX5090 超过 700 tokens/s，最高约 4 倍速度。
- **C3**：26B 命名模型激活约 3.8B，量化后约 18GB 显存目标。
- **C4**：博客明确整体输出质量低于标准 Gemma4，适合交互速度优先的实验任务。

## 争议与不确定点

- Apple Silicon 等统一内存设备未必获得相同优势。
- 博客速度没有代表所有任务；生成质量与采样步数一起变化。

## 关联页面

- 概念：[DiffusionGemma](../concepts/DiffusionGemma.md)
- 概念：[Gemma 4](../concepts/Gemma%204.md)
- 主题：[文本扩散语言模型](../topics/%E6%96%87%E6%9C%AC%E6%89%A9%E6%95%A3%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)

## 这里的术语是什么意思

- **decode**：解码阶段：利用已有输入与生成历史，产生后续输出。
- **MoE**：专家混合：路由器为不同输入选择部分子网络，总容量与每次实际计算不同。

## 方法与实验解读

模型先处理 prompt，再并行修正一个 token block；这样提高单个请求的算术密度。编辑和 infilling 能利用双向上下文，但完成每个 block 仍有 denoising 步数与停止条件。测速度时应记录首 token/block 延迟、完整输出长度、batch 和质量。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Google%20-%202026%20-%20DiffusionGemma%204x%20Faster%20Text%20Generation.md#source-section-1 ) | 保存的发布博客版本。 |
| C2 | [原文]( ../../raw/text/Google%20-%202026%20-%20DiffusionGemma%204x%20Faster%20Text%20Generation.md#source-section-2 ) | 专用 GPU、低并发与作者配置，不能当端到端全设备承诺。 |
| C3 | [原文]( ../../raw/text/Google%20-%202026%20-%20DiffusionGemma%204x%20Faster%20Text%20Generation.md#source-section-2 ) | 量化和运行配置条件，不是仅存激活权重即可部署。 |
| C4 | [原文]( ../../raw/text/Google%20-%202026%20-%20DiffusionGemma%204x%20Faster%20Text%20Generation.md#source-section-2 ) | 速度与质量共同选择，不把 4x 当无代价改善。 |

## 核证范围

核读发布定位、硬件速度、显存与质量取舍。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
