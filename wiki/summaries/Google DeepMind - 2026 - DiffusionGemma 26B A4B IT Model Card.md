---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Google DeepMind - 2026 - DiffusionGemma 26B A4B IT Model Card

## TL;DR（快速导读）

DiffusionGemma 模型卡介绍一个从噪声逐步修正文本块的开放模型，并明确讨论速度收益与质量限制。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

可以把生成过程理解为先形成一段待定文本，再多轮完善；这是生成机制示意，不能直接推断任何任务都更快。

## 来源信息

- 类型：Hugging Face 模型卡 / 官方模型资料
- 原始 HTML：[raw/html/Google DeepMind - 2026 - DiffusionGemma 26B A4B IT Model Card.html](../../raw/html/Google%20DeepMind%20-%202026%20-%20DiffusionGemma%2026B%20A4B%20IT%20Model%20Card.html)
- 全文文本：[raw/text/Google DeepMind - 2026 - DiffusionGemma 26B A4B IT Model Card.md](../../raw/text/Google%20DeepMind%20-%202026%20-%20DiffusionGemma%2026B%20A4B%20IT%20Model%20Card.md)
- 来源 URL：https://huggingface.co/google/diffusiongemma-26B-A4B-it
- 作者：Google DeepMind
- 年份：2026
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

它基于 Gemma 4 专家混合底座，在文本画布上进行多轮去噪。小批量并行生成可以改善单用户速度，但标准 Gemma 4 在多数质量评测上仍更强；应在相同硬件与任务条件下比较。

## 关键事实

- **C1**：规格表为总参数 25.2B、激活 3.8B；128 个专家选 8 个并有 1 个 shared expert。
- **C2**：上下文最高 256K、画布 256 tokens；模态表列 Text/Image，正文把视频解释为帧序列输入。
- **C3**：encoder prefill/KV cache，decoder 对画布做双向 denoising；完成画布后纳入缓存。
- **C4**：默认 Entropy-Bounded Denoising：48 步上限，温度 0.8→0.4，bound0.1，双条件 adaptive stop。
- **C5**：表中多数通用、视觉和长文指标低于 Gemma4 26B A4B，但不是所有指标均更低。

## 争议与不确定点

- 规格表和能力正文的模态措辞不同，本页保留差异并说明视频帧处理。
- 官方表不是本库复跑，速度与质量同时依赖采样参数。

## 关联页面

- 概念：[DiffusionGemma](../concepts/DiffusionGemma.md)
- 概念：[Gemma 4](../concepts/Gemma%204.md)
- 概念：[MoE](../concepts/MoE.md)
- 主题：[文本扩散语言模型](../topics/%E6%96%87%E6%9C%AC%E6%89%A9%E6%95%A3%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)
- [DeepMind](../authors/DeepMind.md)：沿作者或机构继续阅读相关来源。

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **KV cache**：键值缓存：保存已经处理过的位置表示，生成新内容时可复用，避免全部重算。
- **prefill**：提示计算阶段：先处理输入提示，再开始逐步生成输出。
- **benchmark**：评测基准：特定数据、任务与规则的组合，分数只在这些条件下成立。
- **encoder**：编码器：把输入转成模型内部表示。

## 方法与实验解读

模型卡适合核对精确结构和 sampler，发布博客只给速度目标。逐块 denoising 用更多块内并行换掉逐 token 的串行瓶颈，收益在低 batch 更明显。质量比较需核对 instruction-tuned 模型和推荐 EB sampler；上下文上限与 MRCR 实际召回也应分别阅读。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Google%20DeepMind%20-%202026%20-%20DiffusionGemma%2026B%20A4B%20IT%20Model%20Card.md#source-section-2 ) | 26B/A4B 是名称中的近似规模。 |
| C2 | [原文]( ../../raw/text/Google%20DeepMind%20-%202026%20-%20DiffusionGemma%2026B%20A4B%20IT%20Model%20Card.md#source-section-4 ) | 输出为文本，不生成视频或音频。 |
| C3 | [原文]( ../../raw/text/Google%20DeepMind%20-%202026%20-%20DiffusionGemma%2026B%20A4B%20IT%20Model%20Card.md#source-section-1 ) | 整体是块自回归，不能称完全无自回归依赖。 |
| C4 | [原文]( ../../raw/text/Google%20DeepMind%20-%202026%20-%20DiffusionGemma%2026B%20A4B%20IT%20Model%20Card.md#source-section-7 ) | 需随配置保存，不能与任意 sampler 的结果混用。 |
| C5 | [原文]( ../../raw/text/Google%20DeepMind%20-%202026%20-%20DiffusionGemma%2026B%20A4B%20IT%20Model%20Card.md#source-section-3 ) | 例如 HLE no-tools 例外；避免把整体趋势写成逐项全败。 |

## 核证范围

核读 overview、精确参数表、benchmark、能力与采样最佳实践。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
