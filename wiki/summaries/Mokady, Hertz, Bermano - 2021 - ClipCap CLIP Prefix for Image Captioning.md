---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Mokady, Hertz, Bermano - 2021 - ClipCap CLIP Prefix for Image Captioning

## TL;DR（快速导读）

ClipCap 把 CLIP 图像表示映射为语言模型的前缀，让已有视觉和语言模型组合完成图像描述。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

图像描述需要把视觉内容交给文字生成器。ClipCap 用一个映射网络把图像向量转成前缀，再由语言模型生成描述。关注点是模型如何连接、哪些参数更新，以及生成句子是否忠实于图像。

## 具体怎么理解

例如图中有一只趴在沙发上的猫，视觉表示先转成语言模型能接收的前缀，再生成描述。

## 关键事实

- **C1**：映射 CLIP 图像表示到一组 GPT-2 prefix embedding，再生成 caption。
- **C2**：同时研究微调语言模型与冻结语言模型版本，参数与效果分别比较。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Mokady%2C%20Hertz%2C%20Bermano%20-%202021%20-%20ClipCap%20CLIP%20Prefix%20for%20Image%20Captioning.pdf)
- 全文文本：[打开全文文本](../../raw/text/Mokady%2C%20Hertz%2C%20Bermano%20-%202021%20-%20ClipCap%20CLIP%20Prefix%20for%20Image%20Captioning.md)
- 作者：Mokady, Hertz, Bermano
- 年份：2021
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Mokady%2C%20Hertz%2C%20Bermano%20-%202021%20-%20ClipCap%20CLIP%20Prefix%20for%20Image%20Captioning.html)

## 争议与不确定点

- caption 指标与视觉事实准确性不能互换。
- prefix 长度、映射结构和冻结选择影响成本。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无
- [CLIP](../concepts/CLIP.md)：回到相邻方法，核对任务边界。

## 方法与实验解读

ClipCap 让已有视觉表示作为语言模型的连续前缀。它减少重新训练底座的需求，但训练 caption 和映射网络仍必要；生成的流畅描述也可能遗漏或虚构图像细节。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Mokady%2C%20Hertz%2C%20Bermano%20-%202021%20-%20ClipCap%20CLIP%20Prefix%20for%20Image%20Captioning.md#source-section-6 ) | 映射网络是视觉语言连接组件 |
| C2 | [原文]( ../../raw/text/Mokady%2C%20Hertz%2C%20Bermano%20-%202021%20-%20ClipCap%20CLIP%20Prefix%20for%20Image%20Captioning.md#source-section-7 ) | 冻结版本并非所有模型参数完全不训练 |

## 核证范围

核对 §3.1–3.3 的前缀、冻结与映射网络设计及评测指标范围。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
