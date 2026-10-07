---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Google DeepMind - 2026 - Gemini Diffusion

## TL;DR（快速导读）

Gemini Diffusion 实验页展示从噪声反复修正文本的路线，用来理解快速生成与编辑的研究动机。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 来源信息

- 类型：Google DeepMind 模型介绍页
- 原始 HTML：[raw/html/Google DeepMind - 2026 - Gemini Diffusion.html](../../raw/html/Google%20DeepMind%20-%202026%20-%20Gemini%20Diffusion.html)
- 全文文本：[raw/text/Google DeepMind - 2026 - Gemini Diffusion.md](../../raw/text/Google%20DeepMind%20-%202026%20-%20Gemini%20Diffusion.md)
- 来源 URL：https://deepmind.google/models/gemini-diffusion/
- 作者：Google DeepMind
- 年份：2026
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

页面介绍的扩散生成与逐词元自回归生成不同，允许迭代修正一段内容。该来源提供研究展示，不能据此当作可本地部署的开放权重模型；部署信息需看具体模型的发布材料。

## 关键事实

- **C1**：来源将 Gemini Diffusion 定位为实验文本 diffusion 模型/demo。
- **C2**：基准以 pass@1 比较 Gemini2.0 Flash-Lite，未做多数投票；SWE-bench 为单轮非 agent 编辑。
- **C3**：速度为 1479 tokens/s，明确排除 overhead，另外列 0.84s overhead。
- **C4**：表中代码、数学与科学任务优势不一致。

## 争议与不确定点

- 实验 demo 与正式生产 API 的可用性、稳定性不同。
- 能力宣传中的 coherent/control 是研究目标，实际需按任务指标核对。

## 关联页面

- 概念：[DiffusionGemma](../concepts/DiffusionGemma.md)
- 主题：[文本扩散语言模型](../topics/%E6%96%87%E6%9C%AC%E6%89%A9%E6%95%A3%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)
- [DeepMind](../authors/DeepMind.md)：沿作者或机构继续阅读相关来源。

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **benchmark**：评测基准：特定数据、任务与规则的组合，分数只在这些条件下成立。

## 方法与实验解读

实验用反复修正文本替代逐 token 单向生成，速度展示应拆开固定 overhead 与采样时间。输出越短，固定部分越重要；代码局部编辑与知识推理又是不同质量目标。网页可说明技术方向与作者评测，不能替代公开训练配方或独立重现。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Google%20DeepMind%20-%202026%20-%20Gemini%20Diffusion.md#source-section-10 ) | 保存网页时的产品状态，不承诺当前开放入口。 |
| C2 | [原文]( ../../raw/text/Google%20DeepMind%20-%202026%20-%20Gemini%20Diffusion.md#source-section-8 ) | max prompt32K，不能当完整编码代理结果。 |
| C3 | [原文]( ../../raw/text/Google%20DeepMind%20-%202026%20-%20Gemini%20Diffusion.md#source-section-9 ) | 平均采样速度不是用户请求的端到端速度。 |
| C4 | [原文]( ../../raw/text/Google%20DeepMind%20-%202026%20-%20Gemini%20Diffusion.md#source-section-8 ) | 不把快速采样推导为全面高于自回归模型。 |

## 核证范围

核读完整网页的实验定位、评测表脚注和速度定义。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
