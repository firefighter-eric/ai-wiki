---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Black Forest Labs - 2026 - FLUX.2 Overview

## TL;DR（快速导读）

FLUX.2 官方概览将图像生成、编辑和多参考图控制组织成一个家族，供读者按创作需求辨认不同入口。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

用角色参考图和场景文字生成新画面时，应分别检查身份、构图和文字渲染，而不是只看整图美观。

## 来源信息

- 类型：官方文档 / 模型总览
- 来源链接：https://docs.bfl.ai/flux_2/flux2_overview
- 全文文本：../../raw/text/Black Forest Labs - 2026 - FLUX.2 Overview.md
- 作者：Black Forest Labs
- 年份：2026
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

页面关注文本渲染、颜色控制、参考图编辑及不同生成速度与质量需求。它是产品能力概览；做选型时还需查看具体版本、输入限制和实际样例，不能把整个平台的宣传能力套在每个版本上。

## 关键事实

- **C1**：FLUX.2 文档区分 klein/max/pro/flex/dev：速度、质量、可调控制和本地使用各有侧重。
- **C2**：参考图限制按端点不同：klein 最多 4 张；max/pro/flex API 最多 8 张、playground 最多 10 张；dev 推荐不超过 6 张。
- **C3**：max 提供 grounding search；文档还描述颜色、构图和结构化提示控制。
- **C4**：klein 4B 约需 13GB VRAM、Apache 2.0；9B 约 24GB，使用非商用许可并包含 8B Qwen3 文本 embedder。
- **C5**：preview 与固定快照端点使用相同 API 形式，但底层权重可不同；固定端点利于重复生成条件。

## 争议与不确定点

- 耗时与显存会随分辨率、硬件和运行实现改变，文档估计不是保证。
- 价格与入口可能变化，本页不把快照价格当作当前报价。

## 关联页面

- 概念：[FLUX.2](../../wiki/concepts/FLUX.2.md)
- 主题：[扩散模型与文生图](../../wiki/topics/%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)
- [Black Forest Labs](../authors/Black%20Forest%20Labs.md)：沿作者或机构继续阅读相关来源。

## 方法与实验解读

这份概览适合把需求映射到型号：批量预览看小模型，细节与文字控制看 flex，实时信息看 max。选型需同时记录端点、参考数量、显存和许可。产品文档的最佳质量等表述是厂商定位，并未给出统一盲评；本页据此提供功能区分，而不生成跨产品质量榜。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Black%20Forest%20Labs%20-%202026%20-%20FLUX.2%20Overview.md#source-section-3 ) | 所存 2026 年文档快照，不作为永不变化的产品规格。 |
| C2 | [原文]( ../../raw/text/Black%20Forest%20Labs%20-%202026%20-%20FLUX.2%20Overview.md#source-section-3 ) | 不能把最高 10 张泛化到所有型号。 |
| C3 | [原文]( ../../raw/text/Black%20Forest%20Labs%20-%202026%20-%20FLUX.2%20Overview.md#source-section-2 ) | 功能描述；精确色值与身份一致性未在本库独立测量。 |
| C4 | [原文]( ../../raw/text/Black%20Forest%20Labs%20-%202026%20-%20FLUX.2%20Overview.md#source-section-12 ) | 版本、组件与许可分别核对；不把 4B 的许可扩展到 9B。 |
| C5 | [原文]( ../../raw/text/Black%20Forest%20Labs%20-%202026%20-%20FLUX.2%20Overview.md#source-section-14 ) | 即使端点固定，也要保存 seed、输入和其他参数。 |

## 核证范围

核读型号比较、参考图限制、klein 参数和许可表、preview/fixed endpoint 说明。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
