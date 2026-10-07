---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Vidu - 2026 - Pricing

## TL;DR（快速导读）

Vidu 价格页同时列出模型与任务支持，可作为归档时的能力矩阵入口，价格和限制需看具体版本。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 来源信息

- 类型：官方文档 / 定价与能力矩阵
- 原始文件：../../raw/html/Vidu - 2026 - Pricing.html
- 全文文本：../../raw/text/Vidu - 2026 - Pricing.md
- 来源链接：https://platform.vidu.com/docs/pricing
- 作者：Vidu
- 年份：2026
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

页面涉及参考视频生成、续写和同步等能力。这里主要用于辨认哪些接口支持哪些任务，而非推荐实时价格；实际调用应查对应模型、分辨率、素材要求和当时有效的计费条件。

## 关键事实

- **C1**：保存价目表含Q2/Q2-Pro/Q2-Turbo及后来Q3变体，任务/分辨率分别计价。
- **C2**：Q2支持T2V，Q2Pro支持I2V/startend/reference，多档540P/720P/1080P。
- **C3**：img2video/reference2video音频另加15credits；还列audio/timing、lip-sync等接口。

## 争议与不确定点

- 价格和功能会变，本页仅归档记录。
- MotionSync和extension存在不保证每个型号或输入组合都支持。

## 关联页面

- 概念：[Vidu Q2-Pro](../../wiki/concepts/Vidu%20Q2-Pro.md)
- 主题：[视频生成](../../wiki/topics/视频生成.md)
- 作者：[ShengShu Technology](../../wiki/authors/ShengShu%20Technology.md)

## 方法与实验解读

价目表是API产品边界和收费线索，不能证明视频质量或内部结构。保存页已含Q3，阅读Q2时不要把页顶Q3最长时长继承给Q2。具体创作前复核端点参数和现行价格；本页不据历史快照制定预算。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Vidu%20-%202026%20-%20Pricing.md#source-section-1 ) | 动态价目页不是固定技术报告。 |
| C2 | [原文]( ../../raw/text/Vidu%20-%202026%20-%20Pricing.md#source-section-1 ) | 具体表格每个task/model组合。 |
| C3 | [原文]( ../../raw/text/Vidu%20-%202026%20-%20Pricing.md#source-section-1 ) | 附加功能和主任务费用区分。 |

## 核证范围

核读完整型号/任务表与Q2音频附加项，限定历史产品文档。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
