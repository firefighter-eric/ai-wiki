---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Polyak et al. - 2021 - Speech resynthesis from discrete disentangled self-supervised representations

## TL;DR（快速导读）

这篇语音重合成方法分别表示内容、韵律和说话人身份，研究怎样以低码率特征控制合成声音。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

把所有声音特征混在一起，难以只改变其中一项。论文从自监督表示中提取不同信息，再合成语音，并比较重建质量与表示成本。分离程度和控制效果需要通过实验确认。

## 具体怎么理解

例如保留一句话的内容，同时调整节奏或说话人特征；如果表示没有分离，改变一项可能带动其他项。

## 关键事实

- **C1**：分别编码语音内容、F0 与说话人身份，再由解码网络合成波形。
- **C2**：实验区分重建、换说话人、F0 操作与比特率，使用 LJ / VCTK 的 16kHz 数据。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Polyak%20et%20al.%20-%202021%20-%20Speech%20resynthesis%20from%20discrete%20disentangled%20self-supervised%20representations.pdf)
- 全文文本：[打开全文文本](../../raw/text/Polyak%20et%20al.%20-%202021%20-%20Speech%20resynthesis%20from%20discrete%20disentangled%20self-supervised%20representations.md)
- 作者：Polyak et al.
- 年份：2021
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Polyak%20et%20al.%20-%202021%20-%20Speech%20resynthesis%20from%20discrete%20disentangled%20self-supervised%20representations.html)

## 争议与不确定点

- 低比特率与主观质量之间存在取舍。
- 语音重合成不是从文本开始的完整 TTS。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无
- [语音表示、识别与合成](../concepts/%E8%AF%AD%E9%9F%B3%E8%A1%A8%E7%A4%BA%E3%80%81%E8%AF%86%E5%88%AB%E4%B8%8E%E5%90%88%E6%88%90.md)：把本篇方法放到相关任务与比较条件中阅读。

## 方法与实验解读

离散语音单位可作为内容表示，再结合音高和说话人信息重建声音。这样的分工便于控制，但内容编码仍可能带有说话人或韵律信息，不能仅凭模块命名断言完全解耦。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Polyak%20et%20al.%20-%202021%20-%20Speech%20resynthesis%20from%20discrete%20disentangled%20self-supervised%20representations.md#source-section-4 ) | 分开输入控制不等于严格统计独立 |
| C2 | [原文]( ../../raw/text/Polyak%20et%20al.%20-%202021%20-%20Speech%20resynthesis%20from%20discrete%20disentangled%20self-supervised%20representations.md#source-section-7 ) | 音质、可控性与压缩率分别评估 |

## 核证范围

核对 §3 三类编码器、§4 的数据与三种评测设置。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
