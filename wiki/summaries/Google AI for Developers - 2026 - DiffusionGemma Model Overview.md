---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Google AI for Developers - 2026 - DiffusionGemma Model Overview

## TL;DR（快速导读）

DiffusionGemma 官方概览说明输入输出、底座和采样配置，适合先确认模型是否符合自己的部署任务。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

可以把生成过程理解为先形成一段待定文本，再多轮完善；这是生成机制示意，不能直接推断任何任务都更快。

## 来源信息

- 类型：Google AI for Developers 官方文档
- 原始 HTML：[raw/html/Google AI for Developers - 2026 - DiffusionGemma Model Overview.html](../../raw/html/Google%20AI%20for%20Developers%20-%202026%20-%20DiffusionGemma%20Model%20Overview.html)
- 全文文本：[raw/text/Google AI for Developers - 2026 - DiffusionGemma Model Overview.md](../../raw/text/Google%20AI%20for%20Developers%20-%202026%20-%20DiffusionGemma%20Model%20Overview.md)
- 来源 URL：https://ai.google.dev/gemma/docs/diffusiongemma
- 作者：Google AI for Developers
- 年份：2026
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

页面将它描述为以离散扩散输出文本的实验模型，可接收文本、图片和视频输入。生成以一块文本画布为单位并行修正；熵阈值、温度与提前停止等采样设置会影响速度和结果。

## 关键事实

- **C1**：基于 Gemma4 26B/A4B，以 block-autoregressive multi-canvas denoising 生成文本。
- **C2**：支持文本、图片和视频输入，明确不支持音频输入。
- **C3**：速度优势主要面向单加速器、低到中 batch；高 QPS 云端收益减弱。
- **C4**：推荐最大 48 denoising steps，温度 0.8→0.4。
- **C5**：提前停止需画布平均熵低于 0.005 且连续两步预测一致；token 选择 entropy bound 为 0.1。

## 争议与不确定点

- 18GB 显存目标依赖量化和运行开销。
- 高并发服务与本地交互不是同一性能目标，不能直接迁移单请求速度。

## 关联页面

- 概念：[DiffusionGemma](../concepts/DiffusionGemma.md)
- 主题：[文本扩散语言模型](../topics/%E6%96%87%E6%9C%AC%E6%89%A9%E6%95%A3%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **MoE**：专家混合：路由器为不同输入选择部分子网络，总容量与每次实际计算不同。

## 方法与实验解读

encoder 处理并缓存已完成上下文，denoiser 在 256-token 画布中反复更新，完成后追加到缓存。参数控制延迟与输出可靠性，若只记录 tokens/s 而不记录停止策略，将无法复现文档比较。部署可先按推荐值起步，再用目标任务检验质量和响应时间。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Google%20AI%20for%20Developers%20-%202026%20-%20DiffusionGemma%20Model%20Overview.md#source-section-1 ) | 画布之间仍按序列推进，画布内部双向修正。 |
| C2 | [原文]( ../../raw/text/Google%20AI%20for%20Developers%20-%202026%20-%20DiffusionGemma%20Model%20Overview.md#source-section-1 ) | 视频理解与视频生成不同。 |
| C3 | [原文]( ../../raw/text/Google%20AI%20for%20Developers%20-%202026%20-%20DiffusionGemma%20Model%20Overview.md#source-section-2 ) | 文档限定的工作负载。 |
| C4 | [原文]( ../../raw/text/Google%20AI%20for%20Developers%20-%202026%20-%20DiffusionGemma%20Model%20Overview.md#source-section-3 ) | 配置建议而非训练定理或任何任务最优解。 |
| C5 | [原文]( ../../raw/text/Google%20AI%20for%20Developers%20-%202026%20-%20DiffusionGemma%20Model%20Overview.md#source-section-3 ) | 同时满足两个停止条件，未选 token 重新加噪。 |

## 核证范围

核读 overview、tradeoff 与完整 serving configuration 表。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
