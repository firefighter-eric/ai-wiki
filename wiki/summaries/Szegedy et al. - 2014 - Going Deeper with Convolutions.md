---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Szegedy et al. - 2014 - Going Deeper with Convolutions

## TL;DR（快速导读）

Inception 在同一模块中组合不同尺度的分支，并用小投影控制计算，研究预算内的多尺度视觉特征。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Szegedy et al. - 2014 - Going Deeper with Convolutions.pdf
- 原始 HTML：../../raw/html/Szegedy et al. - 2014 - Going Deeper with Convolutions.html
- 全文文本：../../raw/text/Szegedy et al. - 2014 - Going Deeper with Convolutions.md
- 作者：Szegedy et al.
- 年份：2014
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

GoogLeNet 将深度、宽度和分支一起设计，用 1×1 卷积等方式降低部分通道成本。理解时应看分支如何融合、降维如何省计算，以及结构复杂性对实际执行的影响。

## 关键事实

- **C1**：Inception在同一层并行不同kernel与pooling，再拼接表示。
- **C2**：1×1conv用于降维和非线性，控制昂贵大kernel计算。
- **C3**：GoogLeNet使用具体Inception实例，比赛结果含7模型ensemble。

## 争议与不确定点

- 竞赛augment、crop和ensemble都影响分数。
- 模块分支复杂度可能影响设备运行，FLOPs少不保证低延迟。

## 关联页面

- 主题：[经典 CNN 架构](../../wiki/topics/经典%20CNN%20架构.md)
- 主题：[传统 CV](../../wiki/topics/传统%20CV.md)
- 概念：[GoogLeNet](../../wiki/concepts/GoogLeNet.md)

## 方法与实验解读

分支在同一层提供不同尺度感受野，瓶颈先压通道以控制计算。这个思路与VGG只堆小kernel、ResNet解决梯度传递的动机不同，可组合却不能由分类分数判断某一因素唯一有效。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Szegedy%20et%20al.%20-%202014%20-%20Going%20Deeper%20with%20Convolutions.md#source-section-6 ) | 多尺度分支。 |
| C2 | [原文]( ../../raw/text/Szegedy%20et%20al.%20-%202014%20-%20Going%20Deeper%20with%20Convolutions.md#source-section-6 ) | 是否省成本依通道/分支配置。 |
| C3 | [原文]( ../../raw/text/Szegedy%20et%20al.%20-%202014%20-%20Going%20Deeper%20with%20Convolutions.md#source-section-7 ) | 单模型与ensemble不混。 |

## 核证范围

核读motivation、Inception结构、GoogLeNet、训练及竞赛设置。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
