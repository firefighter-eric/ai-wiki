---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Google DeepMind - 2026 - Gemma 4 Model Card

## TL;DR（快速导读）

Gemma 4 模型卡用于核对不同规模与架构的规格，以及长上下文、多模态和工具使用的支持范围。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

一个家族可能有多种架构；使用扩散派生模型时，还要区分底座能力与新的生成接口。

## 来源信息

- 类型：Google AI for Developers 官方模型卡
- 原始 HTML：[raw/html/Google DeepMind - 2026 - Gemma 4 Model Card.html](../../raw/html/Google%20DeepMind%20-%202026%20-%20Gemma%204%20Model%20Card.html)
- 全文文本：[raw/text/Google DeepMind - 2026 - Gemma 4 Model Card.md](../../raw/text/Google%20DeepMind%20-%202026%20-%20Gemma%204%20Model%20Card.md)
- 来源 URL：https://ai.google.dev/gemma/docs/core/model_card_4
- 作者：Google DeepMind
- 年份：2026
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

家族中既有密集模型，也有专家混合模型，能力和运行资源不能按一个名字统一推断。阅读时先选定具体成员，再核对上下文、激活参数和输入输出；DiffusionGemma 使用其中的专家模型作底座。

## 关键事实

- **C1**：保存的后续模型卡含 E2B/E4B/12B Unified/26B A4B/31B；发布博客最初仅列四种。
- **C2**：E2B/E4B 支持 128K，12B/26B/31B 支持 256K；音频仅小模型和12B。
- **C3**：架构交替局部窗口和全局 attention，最后层为全局；全局层采用 unified K/V 与 p-RoPE。
- **C4**：E 小模型用 PLE，effective 参数不计全部 embedding 存储；12B 是 encoder-free unified 输入。
- **C5**：26B A4B 实际25.2B/3.8B，30层、窗口1024，128专家选8加1共享专家。
- **C6**：视频通过帧理解、输出为文本；140+训练语言与35+开箱语言是不同口径。

## 争议与不确定点

- 卡片是官方版本化规格，后续更新可能改变型号范围。
- 工具调用能力不是完整 agent 在真实应用中的成功率。

## 关联页面

- 概念：[Gemma 4](../concepts/Gemma%204.md)
- 概念：[Gemma](../concepts/Gemma.md)
- 概念：[MoE](../concepts/MoE.md)
- 概念：[DiffusionGemma](../concepts/DiffusionGemma.md)
- 主题：[LLM 预训练](../topics/LLM%20预训练.md)
- [DeepMind](../authors/DeepMind.md)：沿作者或机构继续阅读相关来源。

## 这里的术语是什么意思

- **MoE**：专家混合：路由器为不同输入选择部分子网络，总容量与每次实际计算不同。
- **encoder**：编码器：把输入转成模型内部表示。

## 方法与实验解读

家族统一的是交互能力目标，不是所有内部结构和硬件成本。阅读应先确定型号，再用参数、模态和上下文表选择预算；effective/active/total 参数分别衡量查表外的计算、稀疏计算与完整存储。任务成功率仍需以具体模型、thinking 与工具条件评估。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Google%20DeepMind%20-%202026%20-%20Gemma%204%20Model%20Card.md#source-section-1 ) | 不同时间快照分别记载，不能混写发布历史。 |
| C2 | [原文]( ../../raw/text/Google%20DeepMind%20-%202026%20-%20Gemma%204%20Model%20Card.md#source-section-3 ) | 型号能力不同，不将全家族列出的模态扩展到每个型号。 |
| C3 | [原文]( ../../raw/text/Google%20DeepMind%20-%202026%20-%20Gemma%204%20Model%20Card.md#source-section-2 ) | 混合注意力仍保留全局层成本。 |
| C4 | [原文]( ../../raw/text/Google%20DeepMind%20-%202026%20-%20Gemma%204%20Model%20Card.md#source-section-3 ) | 参数命名分母不同。 |
| C5 | [原文]( ../../raw/text/Google%20DeepMind%20-%202026%20-%20Gemma%204%20Model%20Card.md#source-section-4 ) | 精确规格限该变体。 |
| C6 | [原文]( ../../raw/text/Google%20DeepMind%20-%202026%20-%20Gemma%204%20Model%20Card.md#source-section-6 ) | 多语数据存在不保证全部语言同等性能。 |

## 核证范围

核读模型家族、架构和参数表、benchmark 和能力段落。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
